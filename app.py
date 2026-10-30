"""Streamlit app: deteksi & pembacaan plat nomor (YOLO + EasyOCR).

Jalankan: python -m streamlit run app.py

UI langsung tampil; model dimuat di thread latar belakang sehingga user bisa
memilih/mengunggah file sambil menunggu.
"""

import io
import threading

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

import plate_engine as engine  # aman: import berat ditunda di dalam engine

st.set_page_config(page_title="Deteksi Plat Nomor Kendaraan", page_icon="🚗", layout="wide")


# --------------------------------------------------------------------------- #
# Warm-up di latar belakang (dijalankan sekali per proses server)
# --------------------------------------------------------------------------- #
def _warmup() -> None:
    try:
        engine.load_models()
    except Exception:
        pass  # pesan error disimpan di engine.load_error()


@st.cache_resource(show_spinner=False)
def start_warmup() -> threading.Thread:
    thread = threading.Thread(target=_warmup, daemon=True, name="model-warmup")
    thread.start()
    return thread


start_warmup()


# --------------------------------------------------------------------------- #
# Helper
# --------------------------------------------------------------------------- #
def read_image(data: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(data))
    image = ImageOps.exif_transpose(image)  # hormati orientasi EXIF foto HP
    return np.array(image.convert("RGB"))


@st.cache_data(show_spinner=False, max_entries=20)
def run_engine(data: bytes, conf_min: float, fast: bool):
    """Di-cache: mengganti tab/slider yang tidak relevan tidak memproses ulang."""
    return engine.detect_and_read(read_image(data), conf_min=conf_min, fast=fast)


def sidebar_config() -> tuple[float, bool]:
    st.sidebar.header("⚙️ Pengaturan")
    conf_min = st.sidebar.slider(
        "Keyakinan minimum deteksi plat", 0.05, 0.9, 0.3, 0.05,
        help="Turunkan bila plat tidak terdeteksi, naikkan bila banyak salah deteksi.",
    )
    fast = st.sidebar.checkbox(
        "Mode cepat", value=True,
        help="Berhenti begitu ada bacaan yakin & berformat valid. "
             "Matikan untuk mencoba semua varian preprocessing (lebih lambat).",
    )

    st.sidebar.divider()

    def _status() -> None:
        if engine.is_ready():
            st.success("Model siap ✅")
        elif engine.load_error():
            st.error(f"Gagal memuat model:\n\n`{engine.load_error()}`")
        else:
            st.info("⏳ Model dimuat di latar belakang. Kamu sudah bisa unggah "
                    "gambar; hasil muncul begitu model siap.")

    still_loading = not (engine.is_ready() or engine.load_error())
    with st.sidebar:
        st.fragment(_status, run_every=2 if still_loading else None)()

    return conf_min, fast


def show_plate(i: int, plate: engine.PlateResult) -> None:
    st.subheader(f"Plat #{i}")
    if plate.text:
        st.success(f"### 🪪 **{plate.pretty}**")
        if not plate.valid:
            st.warning("Teks terbaca tetapi tidak cocok pola plat Indonesia; "
                       "hasil mungkin kurang akurat.")
    else:
        st.warning("Plat terdeteksi tetapi karakternya tidak terbaca.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Keyakinan OCR", f"{plate.confidence:.2f}")
    c2.metric("Sumber", plate.source)
    c3.metric("Format ID valid", "Ya" if plate.valid else "Tidak")

    i1, i2 = st.columns(2)
    i1.image(plate.crop_rgb, caption="Plat (zoom)", clamp=True)
    i2.image(plate.preprocessed, caption="Setelah preprocessing", clamp=True)

    if plate.candidates:
        with st.expander("Kandidat bacaan"):
            for text, conf, name in plate.candidates:
                st.write(f"`{text}` — keyakinan {conf} — {name}")


def show_result(data: bytes, conf_min: float, fast: bool) -> None:
    if not engine.is_ready() and not engine.load_error():
        st.info("Model masih dimuat, proses akan berjalan otomatis…")
    with st.spinner("Mendeteksi & membaca plat…"):
        try:
            out = run_engine(data, conf_min, fast)
        except Exception as exc:
            st.error(f"Gagal memproses gambar: {type(exc).__name__}: {exc}")
            return

    st.image(out.annotated_rgb, caption="Hasil deteksi", use_container_width=True)
    if not out.plates:
        st.warning(out.message)
        return
    st.caption(out.message)
    for i, plate in enumerate(out.plates, 1):
        st.divider()
        show_plate(i, plate)


# --------------------------------------------------------------------------- #
# Halaman
# --------------------------------------------------------------------------- #
def main() -> None:
    st.title("🚗 Deteksi Plat Nomor Kendaraan")
    st.caption("Alur: foto → deteksi plat (YOLO) → crop + zoom → preprocessing → "
               "OCR (EasyOCR) → validasi format plat Indonesia.")

    conf_min, fast = sidebar_config()
    tab_upload, tab_camera = st.tabs(["📤 Upload Gambar", "📷 Kamera"])

    with tab_upload:
        uploaded = st.file_uploader(
            "Pilih foto kendaraan (JPG/PNG/WEBP)", type=["jpg", "jpeg", "png", "webp"]
        )
        if uploaded is not None:
            show_result(uploaded.getvalue(), conf_min, fast)
        else:
            st.info("Unggah foto kendaraan dengan plat yang terlihat jelas.")

    with tab_camera:
        shot = st.camera_input("Ambil foto plat nomor")
        if shot is not None:
            show_result(shot.getvalue(), conf_min, fast)


if __name__ == "__main__":
    main()