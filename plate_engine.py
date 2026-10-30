"""Mesin deteksi + pembacaan plat nomor (YOLO fast-alpr + EasyOCR).

Pengganti plate_detector.py dan plate_ocr.py. Import library berat (torch,
easyocr, fast_alpr) SENGAJA ditunda sampai load_models() dipanggil, supaya
aplikasi Streamlit bisa tampil seketika.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field

import cv2
import numpy as np

DETECTOR_MODEL = "yolo-v9-t-384-license-plate-end2end"
OCR_MODEL = "cct-xs-v1-global-model"

ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
# 1-2 huruf (wilayah) + 1-4 angka + 0-3 huruf (seri)
PLAT_ID = re.compile(r"^[A-Z]{1,2}\d{1,4}[A-Z]{0,3}$")

# Urutan varian preprocessing: yang paling sering berhasil dicoba dulu.
VARIANT_ORDER = ["sharp", "clahe", "otsu", "otsu_inv", "warna", "sharp_inv"]
EARLY_STOP_SCORE = 1.4  # valid format (+0.6) dan keyakinan >= 0.8


# --------------------------------------------------------------------------- #
# Pemuatan model (lazy, thread-safe)
# --------------------------------------------------------------------------- #
_lock = threading.Lock()
_ready = threading.Event()
_state: dict = {"alpr": None, "reader": None, "error": None}


def is_ready() -> bool:
    return _ready.is_set()


def load_error() -> str | None:
    return _state["error"]


def load_models() -> None:
    """Muat model sekali saja. Aman dipanggil dari banyak thread."""
    if _ready.is_set():
        return
    with _lock:
        if _ready.is_set():
            return
        try:
            import easyocr  # noqa: PLC0415
            import torch  # noqa: PLC0415
            from fast_alpr import ALPR  # noqa: PLC0415

            _state["alpr"] = ALPR(
                detector_model=DETECTOR_MODEL, ocr_model=OCR_MODEL
            )
            _state["reader"] = easyocr.Reader(
                ["en"], gpu=torch.cuda.is_available(), verbose=False
            )
            _state["error"] = None
            _ready.set()
        except Exception as exc:
            _state["error"] = f"{type(exc).__name__}: {exc}"
            raise


# --------------------------------------------------------------------------- #
# Hasil
# --------------------------------------------------------------------------- #
@dataclass
class PlateResult:
    box: tuple[int, int, int, int]
    crop_rgb: np.ndarray
    preprocessed: np.ndarray
    text: str = ""
    pretty: str = ""
    confidence: float = 0.0
    source: str = "-"
    valid: bool = False
    candidates: list = field(default_factory=list)


@dataclass
class EngineOutput:
    annotated_rgb: np.ndarray
    plates: list[PlateResult] = field(default_factory=list)
    message: str = ""
    used_fallback: bool = False


# --------------------------------------------------------------------------- #
# Utilitas teks
# --------------------------------------------------------------------------- #
def bersihkan(teks) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(teks).upper())


def format_plat(teks: str) -> str:
    """B1234XYZ -> B 1234 XYZ"""
    m = re.match(r"^([A-Z]{1,2})(\d{1,4})([A-Z]{0,3})$", teks)
    return " ".join(g for g in m.groups() if g) if m else teks


def skor(teks: str, conf: float) -> float:
    s = conf
    if PLAT_ID.match(teks):
        s += 0.6
    if not (4 <= len(teks) <= 9):
        s -= 0.5
    return s


# --------------------------------------------------------------------------- #
# Gambar
# --------------------------------------------------------------------------- #
def crop_dengan_padding(img, box, pad_x=0.12, pad_y=0.25):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - bw * pad_x))
    x2 = min(w, int(x2 + bw * pad_x))
    y1 = max(0, int(y1 - bh * pad_y))
    y2 = min(h, int(y2 + bh * pad_y))
    return img[y1:y2, x1:x2], (x1, y1, x2, y2)


def zoom(crop, tinggi_target=200):
    h = crop.shape[0]
    skala = max(1.0, tinggi_target / max(h, 1))
    return cv2.resize(crop, None, fx=skala, fy=skala, interpolation=cv2.INTER_CUBIC)


def varian_preprocessing(crop_zoom: np.ndarray) -> dict[str, np.ndarray]:
    gray = cv2.cvtColor(crop_zoom, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    denoise = cv2.fastNlMeansDenoising(clahe, None, h=10)
    blur = cv2.GaussianBlur(denoise, (0, 0), 2)
    sharp = cv2.addWeighted(denoise, 1.6, blur, -0.6, 0)
    _, otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return {
        "warna": crop_zoom,
        "clahe": clahe,
        "sharp": sharp,
        "otsu": otsu,
        "otsu_inv": cv2.bitwise_not(otsu),
        "sharp_inv": cv2.bitwise_not(sharp),
    }


# --------------------------------------------------------------------------- #
# OCR
# --------------------------------------------------------------------------- #
def ocr_easyocr(gambar) -> tuple[str, float]:
    """Baca teks; abaikan tulisan kecil (mis. masa berlaku di bawah plat)."""
    hasil = _state["reader"].readtext(
        gambar, allowlist=ALLOWLIST, detail=1, paragraph=False
    )
    if not hasil:
        return "", 0.0
    tinggi = [abs(b[2][1] - b[0][1]) for b, _, _ in hasil]
    tmax = max(tinggi)
    utama = [(b, t, c) for (b, t, c), th in zip(hasil, tinggi) if th >= 0.5 * tmax]
    utama.sort(key=lambda r: r[0][0][0])
    teks = bersihkan("".join(t for _, t, _ in utama))
    conf = float(np.mean([c for _, _, c in utama]))
    return teks, conf


def baca_plat(crop_zoom, teks_alpr=None, conf_alpr=0.0, fast=True):
    """Kembalikan (teks, conf, sumber, kandidat, citra_preprocessing)."""
    variants = varian_preprocessing(crop_zoom)
    kandidat: list[tuple[float, str, float, str]] = []

    if teks_alpr:
        t = bersihkan(teks_alpr)
        kandidat.append((skor(t, conf_alpr), t, conf_alpr, "fast-alpr"))
        # Mode cepat: bila OCR bawaan sudah yakin & format valid, lewati EasyOCR.
        if fast and kandidat[0][0] >= EARLY_STOP_SCORE:
            kandidat.sort(reverse=True)
            _, t, c, s = kandidat[0]
            return t, c, s, kandidat, variants["sharp"]

    for nama in VARIANT_ORDER:
        t, c = ocr_easyocr(variants[nama])
        if not t:
            continue
        kandidat.append((skor(t, c), t, c, nama))
        if fast and kandidat[-1][0] >= EARLY_STOP_SCORE:
            break

    if not kandidat:
        return "", 0.0, "-", [], variants["sharp"]
    kandidat.sort(reverse=True)
    _, t, c, s = kandidat[0]
    return t, c, s, kandidat, variants["sharp"]


def deteksi_cadangan(img):
    """Bila YOLO gagal: cari teks berpola plat di seluruh gambar."""
    hasil = _state["reader"].readtext(
        img, allowlist=ALLOWLIST, detail=1, paragraph=False
    )
    boxes = []
    for b, t, c in hasil:
        t = bersihkan(t)
        if len(t) >= 4 and PLAT_ID.match(t):
            xs = [p[0] for p in b]
            ys = [p[1] for p in b]
            boxes.append(((int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))), c))
    return boxes


# --------------------------------------------------------------------------- #
# API utama
# --------------------------------------------------------------------------- #
def detect_and_read(
    img_rgb: np.ndarray,
    conf_min: float = 0.3,
    fast: bool = True,
    max_side: int = 1600,
) -> EngineOutput:
    load_models()

    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)  # model dilatih dgn BGR (cv2)
    h, w = img.shape[:2]
    if max(h, w) > max_side:  # foto HP 12MP tidak perlu seukuran penuh
        s = max_side / float(max(h, w))
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

    deteksi = []
    for r in _state["alpr"].predict(img):
        if r.detection.confidence < conf_min:
            continue
        bb = r.detection.bounding_box
        teks = r.ocr.text if r.ocr else None
        conf = r.ocr.confidence if r.ocr else 0.0
        if isinstance(conf, (list, tuple, np.ndarray)):
            conf = float(np.mean(conf)) if len(conf) else 0.0
        deteksi.append(((bb.x1, bb.y1, bb.x2, bb.y2), teks, float(conf)))

    used_fallback = False
    if not deteksi:
        deteksi = [(b, None, 0.0) for b, _ in deteksi_cadangan(img)]
        used_fallback = bool(deteksi)

    annotated = img.copy()
    if not deteksi:
        return EngineOutput(
            annotated_rgb=cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            message="Plat tidak ditemukan. Coba foto yang lebih jelas / lebih dekat.",
        )

    plates: list[PlateResult] = []
    for box, teks_alpr, conf_alpr in deteksi:
        crop, box_pad = crop_dengan_padding(img, box)
        if crop.size == 0:
            continue
        crop_z = zoom(crop)
        teks, conf, sumber, kandidat, pre = baca_plat(
            crop_z, teks_alpr, conf_alpr, fast
        )
        pretty = format_plat(teks) if teks else ""

        x1, y1, x2, y2 = box_pad
        tebal = max(2, img.shape[1] // 400)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), tebal)
        if pretty:
            scale = max(0.6, tebal * 0.35)
            (tw, th), base = cv2.getTextSize(
                pretty, cv2.FONT_HERSHEY_SIMPLEX, scale, tebal
            )
            top = max(y1 - th - base - 6, 0)
            cv2.rectangle(
                annotated, (x1, top), (x1 + tw + 12, top + th + base + 8),
                (0, 200, 0), -1,
            )
            cv2.putText(
                annotated, pretty, (x1 + 6, top + th + 4),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), tebal, cv2.LINE_AA,
            )

        plates.append(
            PlateResult(
                box=box_pad,
                crop_rgb=cv2.cvtColor(crop_z, cv2.COLOR_BGR2RGB),
                preprocessed=pre,
                text=teks,
                pretty=pretty,
                confidence=conf,
                source=sumber,
                valid=bool(PLAT_ID.match(teks)),
                candidates=[(t, round(c, 2), n) for _, t, c, n in kandidat[:4]],
            )
        )

    msg = f"{len(plates)} plat ditemukan."
    if used_fallback:
        msg += " (YOLO tidak menemukan plat, memakai metode cadangan.)"
    return EngineOutput(
        annotated_rgb=cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
        plates=plates,
        message=msg,
        used_fallback=used_fallback,
    )