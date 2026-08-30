# Panduan Setup Python di Laptop Anda

Panduan ini untuk laptop Windows (Axioo, i5-1135G7, 8GB RAM) --
spek ini lebih dari cukup untuk proyek generator SEM ini, karena
tidak ada deep learning atau GPU yang dibutuhkan, cuma operasi
angka biasa (numpy/scipy/pandas).

---

## Langkah 1 -- Install Python

1. Buka browser, kunjungi **https://www.python.org/downloads/**
2. Klik tombol besar "Download Python 3.x.x" (ambil versi terbaru
   yang muncul otomatis).
3. Jalankan file installer yang sudah terunduh.

**Ada 2 kemungkinan tampilan installer, tergantung versi Python:**

**A) Installer klasik (jendela dengan checkbox):**
- **PENTING:** ada checkbox kecil di bagian bawah bertuliskan
  **"Add python.exe to PATH"** -- WAJIB dicentang.
- Klik "Install Now", tunggu sampai selesai.

**B) Python Installation Manager (proses di terminal, bukan jendela
   checkbox -- installer versi lebih baru, misalnya Python 3.14+):**
- Installer akan menanyakan beberapa pertanyaan y/N di terminal,
  misalnya soal "global shortcuts directory" atau "legacy py command".
- Semua pertanyaan ini bersifat **opsional/konfigurasi tambahan** --
  aman dijawab `y` untuk semuanya. Tidak ada yang wajib dijawab
  dengan cara tertentu supaya perintah `python` nanti bisa jalan.
- Kalau diminta restart terminal, **tutup dan buka ulang** jendela
  Command Prompt sebelum lanjut ke langkah berikutnya.

## Langkah 2 -- Pastikan Python terpasang dengan benar

1. Buka **Command Prompt baru** (tekan tombol Windows, ketik "cmd", Enter).
   Kalau installer sebelumnya minta restart terminal, pastikan ini
   benar-benar jendela BARU, bukan yang lama.
2. Ketik:
   ```
   python --version
   ```
3. Kalau muncul tulisan seperti `Python 3.12.x` atau `Python 3.14.x`,
   berarti berhasil. Kalau muncul pesan error "python is not
   recognized":
   - Untuk installer klasik: kemungkinan checkbox "Add to PATH"
     tadi terlewat -- install ulang dan pastikan dicentang.
   - Untuk Python Installation Manager: coba ketik `py --version`
     sebagai gantinya (launcher `py` biasanya tetap terpasang
     meskipun `python` belum dikenali) -- kalau ini berhasil, pakai
     `py` menggantikan `python` di semua perintah pada panduan ini
     (misalnya `py main.py`, bukan `python main.py`).

## Langkah 3 -- Install VS Code (text editor)

1. Kunjungi **https://code.visualstudio.com**
2. Download dan install seperti aplikasi Windows biasa (Next, Next,
   Install).
3. Buka VS Code, masuk ke ikon Extensions (kotak-kotak di sisi kiri),
   cari "Python" (buatan Microsoft), klik Install. Ini yang membuat
   VS Code mengerti kode Python (pewarnaan sintaks, dll).

## Langkah 4 -- Siapkan folder proyek

1. Buat folder baru di komputer Anda, misalnya:
   `C:\Users\NamaAnda\SEM_Generator`
2. Taruh semua file `.py` yang sudah kita buat di folder ini (15 file):
   - `config.py`
   - `latent.py`
   - `structural.py`
   - `indicator.py`
   - `ordinal_engine.py`
   - `quality_control.py`
   - `optimizer.py`
   - `export.py`
   - `report.py`
   - `main.py`
   - `validate_cfa.py`
   - `validate_sem.py`
   - `validate_multigroup.py`
   - `requirements.txt`

   **Penting:** ambil versi PALING BARU dari tiap file (beberapa sudah
   direvisi lebih dari sekali sepanjang percakapan kita -- `config.py`
   khususnya sudah direvisi 5-6 kali). Kalau ragu file mana yang
   terbaru, tanyakan ke saya di chat ini, saya lampirkan ulang.
3. Di VS Code: File > Open Folder... > pilih folder `SEM_Generator`
   tadi. Anda akan melihat semua file muncul di panel kiri.

## Langkah 5 -- Buka terminal di dalam VS Code

1. Menu atas VS Code: Terminal > New Terminal.
2. Akan muncul jendela hitam kecil di bagian bawah -- ini terminal
   yang sudah otomatis berada di folder proyek Anda.

## Langkah 6 -- Install library yang dibutuhkan

Di terminal VS Code tadi, ketik:
```
pip install -r requirements.txt
```
Tunggu sampai selesai (butuh koneksi internet, sekali install saja,
tidak perlu diulang tiap hari).

## Langkah 7 -- Jalankan generatornya

```
python main.py
```
Kalau berhasil, akan muncul tulisan "LOLOS" dan file baru muncul di
folder `output/`: `dataset_sem.csv`, `qc_report.txt`,
`correlation_heatmap.png`.

Untuk validasi tambahan (jalankan setelah `main.py`, urutan bebas):
```
python validate_cfa.py
python validate_sem.py
python validate_multigroup.py
```
- `validate_cfa.py` -- cek model pengukuran (loading, fit index dasar)
- `validate_sem.py` -- cek model struktural gabungan (H1-H6, semua responden)
- `validate_multigroup.py` -- cek moderasi EC (H7a-H7f, Low vs High EC)

## Melihat hasil dataset_sem.csv

File CSV bisa langsung dibuka lewat Excel (klik dua kali di File
Explorer), atau lewat SPSS/AMOS (File > Open > pilih tipe file CSV).

---

## Troubleshooting umum

| Masalah | Kemungkinan penyebab |
|---|---|
| `'python' is not recognized` | Checkbox "Add to PATH" waktu install terlewat -- install ulang Python |
| `pip install` gagal / timeout | Cek koneksi internet, atau coba jaringan lain (kadang WiFi kampus memblokir) |
| `ModuleNotFoundError: No module named 'numpy'` | Berarti langkah 6 (`pip install -r requirements.txt`) belum dijalankan, atau gagal di tengah jalan |
| Terminal VS Code tidak muncul di folder yang benar | Pastikan Anda membuka folder lewat File > Open Folder, bukan File > Open File |
