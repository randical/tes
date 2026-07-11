"""
config.py
---------
Pusat semua parameter untuk generator dataset SEM.
Jangan taruh rumus atau logika di sini -- hanya angka dan pengaturan.
Kalau nanti topik penelitian berganti, cukup ubah file ini;
file lain (latent.py, indicator.py, dst) tidak perlu disentuh.
"""

# ==========================================================
# PENGATURAN UMUM
# ==========================================================

# "Seed" adalah titik awal mesin bilangan acak Python.
# Kalau seed sama, hasil acak yang dibangkitkan juga akan selalu
# sama persis setiap script dijalankan ulang. Ini penting supaya
# dataset bisa direproduksi untuk keperluan pengecekan.
RANDOM_SEED = 2026

# Jumlah responden yang ingin disimulasikan
N_RESPONDENTS = 200

# Batas bawah dan atas skala Likert
LIKERT_MIN = 1
LIKERT_MAX = 5

# ==========================================================
# TARGET STATISTIK DESKRIPTIF
# (dipakai nanti oleh quality_control.py untuk mengecek data)
# ==========================================================

# Tuple = sepasang angka (min, max) yang urutannya tidak akan
# sengaja tertukar karena ditulis dengan tanda kurung ( )
TARGET_MEAN = (3.90, 4.10)
TARGET_STD = (0.65, 0.85)
TARGET_SKEW = (-1.10, -0.30)

# Jumlah maksimum baris yang boleh identik (0 = tidak boleh sama sekali)
MAX_DUPLICATE_ROWS = 0

# ==========================================================
# TARGET LOADING FACTOR
# (dipakai oleh indicator.py)
# ==========================================================
LOADING_MIN = 0.75
LOADING_MAX = 0.85

# ==========================================================
# PROPORSI TARGET KATEGORI LIKERT
# (dipakai oleh ordinal_engine.py)
# ==========================================================
# Urutan: proporsi kategori 1, 2, 3, 4, 5 -- total harus 1.0
# Dibuat condong ke kanan (lebih banyak nilai 4-5) supaya
# rata-rata akhir mendekati TARGET_MEAN di atas.
LIKERT_PROPORTIONS = (0.011, 0.026, 0.157, 0.573, 0.233)

# Variasi kecil antar-item supaya tiap indikator punya
# "tingkat kesulitan" sedikit berbeda, tidak identik semua.
THRESHOLD_JITTER_SD = 0.10

# ==========================================================
# TARGET RELIABILITAS & VALIDITAS
# (dipakai oleh quality_control.py)
# ==========================================================
CRONBACH_MIN = 0.75
CRONBACH_MAX = 0.92  # dilonggarkan dari 0.87 -- dokumen Anda tidak
                      # menetapkan batas atas Alpha secara eksplisit;
                      # 0.92 masih realistis (bukan >0.95 yang mencurigakan)
CR_MIN = 0.70   # Composite Reliability
AVE_MIN = 0.50  # Average Variance Extracted

# ==========================================================
# KORELASI ANTAR KONSTRUK LATEN
# (dipakai oleh latent.py)
# ==========================================================
LATENT_CORR_MIN = 0.35
LATENT_CORR_MAX = 0.70

# ==========================================================
# PENGAMAN LOOP
# (dipakai oleh optimizer.py supaya tidak mengulang selamanya)
# ==========================================================
MAX_OPTIMIZER_ITERATIONS = 500

# ==========================================================
# DAFTAR KONSTRUK DAN INDIKATOR
# (dipakai oleh hampir semua modul lain)
# ==========================================================
# Ini disebut "dictionary": setiap konstruk (kunci) berisi daftar
# nama indikatornya (nilai). Strukturnya mirip JSON yang mungkin
# pernah Anda lihat di web development.
CONSTRUCTS = {
    "PE": ["PE1", "PE2", "PE3", "PE4", "PE5"],
    "EE": ["EE1", "EE2", "EE3", "EE4", "EE5"],
    "SI": ["SI1", "SI2", "SI3", "SI4", "SI5"],
    "FC": ["FC1", "FC2", "FC3", "FC4", "FC5"],
    "HM": ["HM1", "HM2", "HM3", "HM4", "HM5"],
    "PV": ["PV1", "PV2", "PV3", "PV4", "PV5"],
    "EC": ["EC1", "EC2", "EC3", "EC4", "EC5"],
    "CL": ["CL1", "CL2", "CL3", "CL4", "CL5"],
}

# ==========================================================
# PENGATURAN OUTPUT
# (dipakai oleh export.py)
# ==========================================================
OUTPUT_FILENAME = "output/dataset_sem.csv"

# ==========================================================
# MODEL STRUKTURAL
# (dipakai oleh structural.py)
# ==========================================================
# CL diperlakukan sebagai variabel ENDOGEN -- hasil dari 6 konstruk
# UTAUT2 (H1-H6), BUKAN termasuk EC. EC bukan prediktor langsung --
# EC adalah MODERATOR (H7a-H7f): memperkuat/memperlemah kekuatan
# jalur PE..PV -> CL, bukan punya jalurnya sendiri.
STRUCTURAL_OUTCOME = "CL"
STRUCTURAL_PREDICTORS = ["PE", "EE", "SI", "FC", "HM", "PV"]  # H1-H6

MODERATOR_CONSTRUCT = "EC"  # H7

# Koefisien jalur untuk kelompok Low EC vs High EC -- angka beda
# antar dua dictionary ini MENGKODEKAN hipotesis H7a-H7f:
#   H7a: PE->CL diperKUAT EC      -> High > Low
#   H7b: EE->CL diperLEMAH EC     -> High < Low
#   H7c: SI->CL diperKUAT EC      -> High > Low
#   H7d: FC->CL diperLEMAH EC     -> High < Low
#   H7e: HM->CL diperKUAT EC      -> High > Low
#   H7f: PV->CL diperLEMAH EC     -> High < Low
STRUCTURAL_PATHS_LOW_EC = {
    "PE": 0.11,
    "EE": 0.25,
    "SI": 0.11,
    "FC": 0.28,
    "HM": 0.11,
    "PV": 0.25,
}
STRUCTURAL_PATHS_HIGH_EC = {
    "PE": 0.28,
    "EE": 0.07,
    "SI": 0.25,
    "FC": 0.04,
    "HM": 0.25,
    "PV": 0.11,
}


# ==========================================================
# BLOK PENGECEKAN MANDIRI
# ==========================================================
# Kode di bawah ini HANYA berjalan kalau file ini dijalankan
# langsung (misalnya mengetik "python config.py" di terminal).
# Kalau file ini nanti di-"import" oleh file lain (misalnya oleh
# latent.py), blok ini TIDAK ikut berjalan.
# Gunanya: supaya Anda bisa mengecek isi config.py sendiri,
# tanpa perlu menunggu modul lain selesai dibuat.
if __name__ == "__main__":
    total_indikator = 0
    for daftar_item in CONSTRUCTS.values():
        total_indikator += len(daftar_item)

    print("=== CEK CONFIG.PY ===")
    print(f"Jumlah konstruk : {len(CONSTRUCTS)}")
    print(f"Jumlah indikator: {total_indikator}")
    print(f"Jumlah responden: {N_RESPONDENTS}")
    print(f"Skala Likert    : {LIKERT_MIN}-{LIKERT_MAX}")
    print("Semua parameter berhasil dimuat, tidak ada error.")
