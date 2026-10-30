# 🚗 Deteksi Plat Nomor Kendaraan (Streamlit)

Aplikasi web untuk **mendeteksi lokasi plat nomor** pada foto kendaraan dan
**membaca nomornya** (mis. `B 1234 XYZ`). Dibuat dengan Python + OpenCV
(classical computer vision) + OCR, siap di-deploy ke Streamlit Community Cloud.

Proyek ini adalah hasil *breakdown* notebook `Untitled13.ipynb` (pipeline
deteksi tumor otak MRI) yang dialihkan fungsinya ke deteksi plat nomor.

---

## 1. Cara menjalankan

### a. Install dependensi Python

```bash
pip install -r requirements.txt
```

### b. Install mesin OCR (Tesseract)

`pytesseract` hanya pembungkus; binary Tesseract harus ada di sistem.

| OS | Perintah |
|----|----------|
| Windows | Installer: <https://github.com/UB-Mannheim/tesseract/wiki> (centang *Add to PATH*), atau `choco install tesseract` |
| Ubuntu/Debian | `sudo apt install tesseract-ocr` |
| macOS | `brew install tesseract` |
| Google Colab | `!apt install tesseract-ocr` lalu `!pip install pytesseract` |

Cek: `tesseract --version`.

Jika Tesseract tidak ingin dipasang, gunakan backend alternatif:

```bash
pip install easyocr   # lebih berat (butuh torch), model diunduh saat pertama jalan
```

### c. Jalankan aplikasi

```bash
streamlit run app.py
```

Buka <http://localhost:8501>, lalu unggah foto atau ambil foto lewat kamera.

---

## 2. Cara deploy ke Streamlit Cloud (gratis)

1. Push folder ini ke repository GitHub (isi: `app.py`, `plate_detector.py`,
   `plate_ocr.py`, `requirements.txt`, `packages.txt`, `README.md`).
2. Buka <https://share.streamlit.io> → **New app**.
3. Pilih repository, branch, dan **Main file path: `app.py`**.
4. Klik **Deploy**. Streamlit akan:
   - memasang paket di `requirements.txt`,
   - memasang `tesseract-ocr` dari `packages.txt` (apt),
   - menjalankan aplikasi.
5. Selesai — URL publik siap dibagikan.

Catatan:
- `packages.txt` **wajib** ada, kalau tidak OCR akan error
  `tesseract is not installed or it's not in your PATH`.
- `opencv-python-headless` dipakai (bukan `opencv-python`) karena server tidak
  punya GUI.
- `easyocr` di Cloud butuh memori besar dan unduhan model; tesseract lebih
  ringan dan lebih cepat start.

---

## 3. Breakdown notebook → aplikasi

Alur notebook asli dan padanannya di kode ini:

| Notebook (`Untitled13.ipynb`) | Di aplikasi | Lokasi kode |
|---|---|---|
| `cv2.imread` | upload / kamera Streamlit → array RGB | `app.py: read_image()` |
| `cv2.cvtColor(BGR2RGB)` + `BGR2GRAY` | grayscale (Pert. 1) | `plate_detector.py: _to_gray()` |
| `cv2.GaussianBlur(gray, (5,5), 0)` | reduksi noise (Pert. 5) | `_denoise()` |
| `cv2.threshold(blur, 180, 255, THRESH_BINARY)` | threshold / Canny / Sobel | `_edges()` |
| `cv2.findContours(..., RETR_TREE, ...)` | `RETR_EXTERNAL` + filter bentuk | `_candidates()` |
| `cv2.drawContours(..., -1, (0,255,0), 2)` | kotak plat + label hasil OCR | `annotate()` |
| `plt.imshow(...)` banyak subplot | gambar tiap tahap di Streamlit | `show_pipeline()` |
| *(baru)* | potong & luruskan plat | `four_point_transform()` |
| *(baru)* | OCR + normalisasi teks | `plate_ocr.py: read_plate()` |

Perbedaan penting dari notebook:

1. **RETR_TREE → RETR_EXTERNAL.** Notebook menggambar semua kontur (tengkorak
   dan tumor). Untuk plat kita hanya butuh kontur terluar (blok plat).
2. **Morphological closing.** Karakter pada plat terpisah-pisah, jadi tepi
   ditutup dengan kernel lebar (default 17×3) agar menyatu menjadi satu blok
   persegi panjang yang bisa dideteksi.
3. **Filter bentuk.** Kandidat disaring dengan rasio aspek (default 2.0–6.5,
   ideal 3.0), luas minimum, dan kemiringan maksimum, lalu diberi skor.
4. **Perspective warp.** Plat difoto miring/keperpektif, jadi 4 titik sudut
   dipetakan ke persegi menggunakan `cv2.getPerspectiveTransform`, sama seperti
   *four point transform* pada modul OCR populer.
5. **Multi-varian OCR.** Citra plat dibuat beberapa versi (grayscale tajam,
   Otsu, Otsu terbalik, threshold tetap, adaptive). Setiap varian di-OCR dan
   yang skornya terbaik dipakai — jauh lebih tahan terhadap bayangan.

---

## 4. Struktur file

```
app.py             # UI Streamlit (upload, kamera, slider parameter, hasil)
plate_detector.py  # pipeline OpenCV: grayscale → blur → tepi → kontur → warp
plate_ocr.py       # OCR (pytesseract/easyocr) + normalisasi "B 1234 XYZ"
requirements.txt   # dependensi pip
packages.txt       # paket apt untuk Streamlit Cloud (tesseract-ocr)
tests/test_pipeline.py  # smoke test pipeline + unit test normalisasi teks
```

Modul deteksi dan OCR tidak bergantung pada Streamlit, jadi bisa dipakai di
script/notebook lain:

```python
import cv2
from plate_detector import DetectorConfig, find_plate
from plate_ocr import OcrConfig, read_plate

image = cv2.cvtColor(cv2.imread("mobil.jpg"), cv2.COLOR_BGR2RGB)
detection = find_plate(image, DetectorConfig())
ocr = read_plate(detection.variants, OcrConfig())
print(ocr.text, ocr.valid, ocr.confidence)
```

---

## 5. Batasan

- Metode klasik peka terhadap kontras, sudut, dan ukuran plat. Plat yang sangat
  miring, kotor, atau gelap bisa gagal terdeteksi.
- OCR Tesseract akurat untuk plat cetak yang tajam. Huruf/angka mirip
  (`0/O`, `1/I`, `8/B`, `5/S`) sudah dikoreksi lewat aturan posisi, tapi tetap
  bisa salah.
- Untuk akurasi produksi, ganti tahap deteksi dengan model deep learning
  (YOLOv8 khusus plat) dan OCR dengan model khusus plat Indonesia; antarmuka
  Streamlit dan modul OCR tetap bisa dipakai.

## 6. Referensi

- Dokumentasi OpenCV (contours, morphology, perspective transform)
- `pytesseract` – page segmentation mode 7 (satu baris teks)
- Streamlit Community Cloud – `requirements.txt` + `packages.txt`
