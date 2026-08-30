"""
finalisasi.py
-------------
Merapikan dataset akhir.

  1. Menambahkan kolom kode angka untuk SELURUH demografi. AMOS membaca
     angka, bukan teks, jadi kolom ini wajib ada kalau demografi mau
     dipakai sebagai variabel pengelompok maupun variabel kontrol.
  2. Urutan kode mengikuti URUTAN DI KUESIONER, bukan urutan internal
     generator. Ini penting supaya tabel di naskah cocok dengan
     instrumennya.
  3. Menulis kodebuk dan sintaks SPSS.
"""
import numpy as np
import pandas as pd
import config

# Urutan kode mengikuti Bagian A kuesioner.
KODE = {
    "Domisili": ["Kota Pontianak", "Kabupaten Kubu Raya", "Kota Singkawang",
                 "Kabupaten Mempawah", "Kabupaten Sambas",
                 "Kabupaten Bengkayang", "Kabupaten Landak",
                 "Kabupaten Sanggau", "Kabupaten Sekadau",
                 "Kabupaten Sintang", "Kabupaten Melawi",
                 "Kabupaten Kapuas Hulu", "Kabupaten Ketapang",
                 "Kabupaten Kayong Utara"],
    "Gender": ["Laki-laki", "Perempuan"],
    "Usia": ["< 20 tahun", "21-30 tahun", "31-40 tahun", "41-50 tahun",
             "> 50 tahun"],
    "Pendidikan": ["SMA/SMK", "D3", "S1", "S2", "S3"],
    "Pekerjaan": ["Pelajar/Mahasiswa", "ASN/TNI/POLRI", "Pegawai Swasta",
                  "Wiraswasta", "BUMN/BUMD", "Ibu Rumah Tangga", "Lainnya"],
    "Pendapatan": ["< Rp 3.000.000", "Rp 3.000.000 - Rp 5.000.000",
                   "Rp 5.000.001 - Rp 10.000.000",
                   "Rp 10.000.001 - Rp 15.000.000", "> Rp 15.000.000"],
    "Kendaraan": ["Tidak memiliki kendaraan", "Sepeda motor", "Mobil",
                  "Sepeda motor dan mobil"],
    "PernahPakaiEV": ["Ya", "Tidak"],
}

ITEM_TEKS = {
    "PEOU1": "Kendaraan listrik mudah dipelajari cara penggunaannya",
    "PEOU2": "Kendaraan listrik mudah dioperasikan dalam aktivitas sehari-hari",
    "PEOU3": "Proses pengisian daya kendaraan listrik mudah dilakukan",
    "PEOU4": "Fitur-fitur kendaraan listrik mudah dipahami",
    "PEOU5": "Yakin akan mudah beradaptasi jika menggunakan kendaraan listrik",
    "PU1": "Menghemat biaya operasional kendaraan",
    "PU2": "Mengurangi pengeluaran bahan bakar",
    "PU3": "Meningkatkan efisiensi mobilitas sehari-hari",
    "PU4": "Memberikan manfaat jangka panjang bagi penggunanya",
    "PU5": "Lebih ramah lingkungan dibanding kendaraan konvensional",
    "PR1": "Khawatir harga kendaraan listrik masih terlalu mahal",
    "PR2": "Khawatir biaya penggantian baterai mahal",
    "PR3": "Khawatir keterbatasan jarak tempuh",
    "PR4": "Khawatir performa belum sebaik kendaraan konvensional",
    "PR5": "Khawatir keamanan baterai",
    "PR6": "Khawatir jumlah SPKLU di Kalimantan Barat masih terbatas",
    "PR7": "Khawatir pandangan orang sekitar kurang mendukung",
    "GP1": "Informasi program pemerintah mudah diperoleh",
    "GP2": "Pemerintah aktif mempromosikan penggunaan kendaraan listrik",
    "GP3": "Subsidi pemerintah menarik minat terhadap kendaraan listrik",
    "GP4": "Pemerintah aktif menyediakan infrastruktur SPKLU",
    "GP5": "Kebijakan pemerintah mendukung penggunaan kendaraan listrik",
    "AI1": "Berniat menggunakan kendaraan listrik di masa depan",
    "AI2": "Berencana membeli kendaraan listrik dalam beberapa tahun ke depan",
    "AI3": "Bersedia beralih dari kendaraan konvensional ke kendaraan listrik",
    "AI4": "Bersedia merekomendasikan kendaraan listrik kepada orang lain",
    "AI5": "Berminat mencari informasi lebih lanjut mengenai kendaraan listrik",
    "AI6": "Kendaraan listrik salah satu pilihan utama yang dipertimbangkan",
}

NAMA_KONSTRUK = {
    "PEOU": "Perceived Ease of Use", "PU": "Perceived Usefulness",
    "PR": "Perceived Risk", "GP": "Government Promotion",
    "AI": "Adoption Intention",
}

df = pd.read_csv(config.OUTPUT_FILENAME)

# --- kode angka untuk seluruh demografi ---
if "Gender_kode" in df.columns:
    df = df.drop(columns=["Gender_kode"])
for kol, labels in KODE.items():
    peta = {l: i + 1 for i, l in enumerate(labels)}
    belum = set(df[kol].unique()) - set(peta)
    assert not belum, f"Label tak terpetakan pada {kol}: {belum}"
    df[kol + "_kode"] = df[kol].map(peta).astype(int)

urut = (["ID"] + list(KODE) + [k + "_kode" for k in KODE]
        + config.all_items() + [f"Mean_{k}" for k in config.CONSTRUCTS])
df = df[urut]
df.to_csv(config.OUTPUT_FILENAME, index=False)

# ==========================================================
# KODEBUK
# ==========================================================
K = ["# KODEBUK DATASET SINTETIS",
     "",
     "**Proyek.** TAM Analysis, Perceived Risk, Government Promotion terhadap",
     "Adoption Intention Kendaraan Listrik melalui Demografi sebagai Variabel",
     "Moderating di Kalimantan Barat.",
     "",
     "**Berkas.** `dataset_EV_Kalbar_SINTETIS.csv`, 200 baris, "
     f"{df.shape[1]} kolom.",
     "",
     "> **PERINGATAN.** Dataset ini sintetis. Angkanya tidak berasal dari",
     "> responden mana pun. Sah dipakai untuk melatih alur SPSS dan AMOS.",
     "> Tidak boleh dilaporkan sebagai data responden dalam karya ilmiah.",
     "",
     "---",
     "",
     "## 1. Kolom identitas",
     "",
     "| Kolom | Isi |",
     "|---|---|",
     "| `ID` | Nomor urut responden, 1 sampai 200 |",
     "",
     "## 2. Kolom demografi",
     "",
     "Setiap variabel hadir dalam dua bentuk. Kolom berlabel teks untuk",
     "menyusun tabel profil di SPSS, dan kolom berakhiran `_kode` berisi",
     "angka untuk AMOS. Urutan kode mengikuti urutan pilihan di kuesioner.",
     ""]
for kol, labels in KODE.items():
    K.append(f"### {kol}")
    K.append("")
    K.append("| Kode | Label | Jumlah |")
    K.append("|---|---|---|")
    vc = df[kol].value_counts()
    for i, l in enumerate(labels):
        K.append(f"| {i + 1} | {l} | {int(vc.get(l, 0))} |")
    K.append("")

K += ["## 3. Kolom indikator", "",
      "Seluruh item memakai skala Likert 5 titik. "
      "1 Sangat Tidak Setuju sampai 5 Sangat Setuju.", "",
      "| Kolom | Konstruk | Ringkasan pernyataan |", "|---|---|---|"]
for k, its in config.CONSTRUCTS.items():
    for it in its:
        K.append(f"| `{it}` | {k} | {ITEM_TEKS[it]} |")

K += ["", "## 4. Kolom skor komposit", "",
      "Rata-rata item per konstruk. Berguna untuk analisis pendahuluan di",
      "SPSS. **Jangan** dipakai sebagai pengganti model laten di AMOS,",
      "karena skor komposit masih memuat error pengukuran.", "",
      "| Kolom | Konstruk | Jumlah item |", "|---|---|---|"]
for k, its in config.CONSTRUCTS.items():
    K.append(f"| `Mean_{k}` | {NAMA_KONSTRUK[k]} | {len(its)} |")

K += ["", "---", "",
      "## 5. Cara memakai di AMOS", "",
      "1. Buka CSV di SPSS, simpan sebagai `.sav`.",
      "2. Gambar model pengukuran lima konstruk, jalankan CFA lebih dulu.",
      "3. Setelah CFA memadai, tambahkan empat panah berarah ke Adoption",
      "   Intention untuk menguji H1 sampai H4.",
      "4. Untuk H5 sampai H8, pakai **Manage Groups** dengan variabel",
      "   pengelompok `Gender_kode`, lalu **Multiple-Group Analysis**.",
      "5. Uji resmi tiap hipotesis moderasi adalah **Delta chi-square** antara",
      "   model yang jalurnya dikekang setara dan model yang dibebaskan,",
      "   dengan derajat bebas satu.",
      "",
      "**Peringatan.** Jangan menyimpulkan moderasi hanya dari perbedaan pola",
      "signifikansi antar kelompok. Itu kesalahan yang tercatat di basis",
      "pengetahuan proyek ini, di mana lima jalur pernah dinyatakan termoderasi",
      "padahal setelah uji formal hanya tiga yang bertahan.", ""]

open("output/KODEBUK.md", "w").write("\n".join(K))

# ==========================================================
# SINTAKS SPSS
# ==========================================================
S = ["* Sintaks SPSS untuk dataset sintetis EV Kalimantan Barat.",
     "* Dataset ini SINTETIS. Bukan data responden.",
     "* Jalankan setelah membaca CSV lewat File > Import Data > CSV Data.",
     ""]
S.append("VARIABLE LABELS")
for kol in KODE:
    S.append(f"  {kol}_kode '{kol}'")
for k, its in config.CONSTRUCTS.items():
    for it in its:
        S.append(f"  {it} '{k} - {ITEM_TEKS[it]}'")
for k in config.CONSTRUCTS:
    S.append(f"  Mean_{k} '{NAMA_KONSTRUK[k]} (skor komposit)'")
S[-1] = S[-1] + "."
S.append("")
S.append("VALUE LABELS")
for kol, labels in KODE.items():
    baris = f"  /{kol}_kode " + " ".join(
        f"{i + 1} '{l}'" for i, l in enumerate(labels))
    S.append(baris)
S.append("  /PEOU1 TO AI6 1 'Sangat Tidak Setuju' 2 'Tidak Setuju' "
         "3 'Netral' 4 'Setuju' 5 'Sangat Setuju'.")
S += ["", "EXECUTE.", "",
      "* Reliabilitas per konstruk."]
for k, its in config.CONSTRUCTS.items():
    S.append(f"RELIABILITY /VARIABLES={' '.join(its)} /SCALE('{k}') ALL "
             "/MODEL=ALPHA /STATISTICS=DESCRIPTIVE SCALE CORR "
             "/SUMMARY=TOTAL.")
S += ["", "* Profil responden.",
      "FREQUENCIES VARIABLES=" + " ".join(f"{k}_kode" for k in KODE) + ".",
      "",
      "* Tabulasi silang untuk memeriksa konsistensi.",
      "CROSSTABS /TABLES=Pekerjaan_kode BY Pendapatan_kode.",
      "CROSSTABS /TABLES=Usia_kode BY Pendidikan_kode.",
      "",
      "* Simpan sebagai .sav sebelum dibuka di AMOS.",
      "* SAVE OUTFILE='dataset_EV_Kalbar_SINTETIS.sav'.", ""]

open("output/sintaks_spss.sps", "w").write("\n".join(S))

print("Kolom akhir:", df.shape[1])
print(list(df.columns))
print("\nKodebuk dan sintaks SPSS ditulis.")
