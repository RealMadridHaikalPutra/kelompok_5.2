"""Smoke test pipeline + unit test normalisasi teks plat.

Jalankan: python tests/test_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plate_detector import (  # noqa: E402
    DetectorConfig,
    find_plate,
    four_point_transform,
    order_points,
)
from plate_ocr import (  # noqa: E402
    available_backends,
    is_valid_plate,
    normalize_plate_text,
    read_plate,
)


def make_scene() -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Buat foto sintetis: plat putih miring berisi teks di atas latar gelap."""
    plate = np.full((130, 400), 245, dtype=np.uint8)
    cv2.putText(
        plate, "B 1234 XYZ", (12, 95), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 20, 5, cv2.LINE_AA
    )
    plate = cv2.copyMakeBorder(plate, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=25)

    canvas = np.full((600, 900, 3), 55, dtype=np.uint8)
    canvas[:, :] = (55, 58, 62)
    noise = np.random.default_rng(0).normal(0, 6, canvas.shape).astype(np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    patch = cv2.cvtColor(plate, cv2.COLOR_GRAY2BGR)
    matrix = cv2.getRotationMatrix2D((patch.shape[1] / 2, patch.shape[0] / 2), 4.0, 1.0)
    rotated = cv2.warpAffine(
        patch, matrix, (patch.shape[1], patch.shape[0]), borderValue=(25, 25, 25)
    )

    y, x = 320, 240
    canvas[y : y + rotated.shape[0], x : x + rotated.shape[1]] = rotated
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return rgb, (x, y, rotated.shape[1], rotated.shape[0])


def test_detection() -> None:
    image, truth = make_scene()
    detection = find_plate(image, DetectorConfig())

    assert detection.found, f"plat sintetis tidak terdeteksi: {detection.message}"
    assert detection.bbox is not None
    assert detection.plate_rgb is not None and detection.plate_rgb.size > 0

    x1, y1, w1, h1 = detection.bbox
    x2, y2, w2, h2 = truth
    cx1, cy1 = x1 + w1 / 2, y1 + h1 / 2
    assert x2 - 20 <= cx1 <= x2 + w2 + 20, f"pusat bbox X salah: {cx1} vs {truth}"
    assert y2 - 20 <= cy1 <= y2 + h2 + 20, f"pusat bbox Y salah: {cy1} vs {truth}"

    crop_h, crop_w = detection.plate_rgb.shape[:2]
    aspect = crop_w / crop_h
    assert 2.0 <= aspect <= 4.5, f"rasio aspek crop aneh: {aspect:.2f}"

    expected = {"gray", "otsu", "otsu_inv", "otsu_fixed", "adaptive"}
    assert expected == set(detection.variants), set(detection.variants)
    assert "Plat (perspective warp)" in detection.debug


def test_geometry_helpers() -> None:
    pts = np.array([[0, 0], [200, 0], [200, 80], [0, 80]], dtype=np.float32)
    ordered = order_points(pts[[2, 0, 3, 1]])
    assert ordered[0].tolist() == [0, 0]
    assert ordered[2].tolist() == [200, 80]

    image = np.zeros((200, 300, 3), dtype=np.uint8)
    image[10:90, 20:220] = 255
    warped = four_point_transform(
        image, np.array([[20, 10], [220, 10], [220, 90], [20, 90]], dtype=np.float32)
    )
    assert warped.shape[1] == 200 and warped.shape[0] == 80
    assert warped.mean() > 200


def test_normalization() -> None:
    cases = {
        "B 1234 XYZ": "B 1234 XYZ",
        "b1234xyz": "B 1234 XYZ",
        "81234XYZ": "B 1234 XYZ",      # 8 -> B pada kelompok huruf depan
        "B 12S4 XYZ": "B 1254 XYZ",   # S di tengah angka -> 5
        "B1234XY2": "B 1234 XYZ",     # 2 di ujung -> Z
        "B 1234": "B 1234",
        "  AB 12 CD  ": "AB 12 CD",
        "--B.1.2.3.4.X.Y.Z--": "B 1234 XYZ",
    }
    for raw, expected in cases.items():
        got = normalize_plate_text(raw)
        assert got == expected, f"normalize({raw!r}) = {got!r}, harusnya {expected!r}"

    assert normalize_plate_text("") == ""
    assert is_valid_plate("B 1234 XYZ")
    assert is_valid_plate("AB12CD")
    assert not is_valid_plate("1234")


def main() -> None:
    test_detection()
    print("OK  deteksi plat + perspective warp")
    test_geometry_helpers()
    print("OK  helper geometri")
    test_normalization()
    print("OK  normalisasi teks plat")

    backends = available_backends()
    print("Backend OCR tersedia:", backends or "TIDAK ADA")
    if backends:
        image, _ = make_scene()
        detection = find_plate(image, DetectorConfig())
        ocr = read_plate(detection.variants)
        print(f"OCR -> text={ocr.text!r} valid={ocr.valid} "
              f"conf={ocr.confidence:.1f} backend={ocr.backend}/{ocr.variant}")
    else:
        print("Lewati uji OCR: Tesseract/easyocr tidak terpasang di mesin ini.")


if __name__ == "__main__":
    main()
