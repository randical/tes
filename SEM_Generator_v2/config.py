"""
config.py
---------
Pusat SEMUA parameter dan deklarasi model. Tidak ada logika di sini,
hanya angka dan struktur data.

Analogi web: file ini seperti file JSON pengaturan (atau blok :root
berisi CSS custom properties). Modul lain tinggal membacanya.

Kalau topik penelitian berganti, HANYA file ini yang disentuh.
"""

# ==========================================================
# BAGIAN A -- PENGATURAN UMUM
# ==========================================================

RANDOM_SEED = 2026
N_RESPONDENTS = 400

# Jumlah titik skala. 5 dan 7 sudah punya proporsi terkalibrasi
# di bagian D. Angka lain butuh pencarian numerik ulang
# (lihat ordinal_engine.find_proportions).
SCALE_POINTS = 5

# ==========================================================
# BAGIAN B -- DEFINISI KONSTRUK DAN INDIKATOR
# ==========================================================
# Kunci   = nama konstruk laten (jadi nama lingkaran di AMOS)
# Nilai   = daftar nama kolom indikator (jadi nama kotak di AMOS)

CONSTRUCTS = {
    "SI":  ["SI1", "SI2", "SI3", "SI4", "SI5"],   # Social Influence
    "PN":  ["PN1", "PN2", "PN3", "PN4", "PN5"],   # Personal Norm
    "AP":  ["AP1", "AP2", "AP3", "AP4"],          # Anticipated Pride
    "CIA": ["CIA1", "CIA2", "CIA3", "CIA4", "CIA5"],  # Charging Infra Anxiety
    "PE":  ["PE1", "PE2", "PE3", "PE4"],          # Performance Expectancy (kontrol)
    "PI":  ["PI1", "PI2", "PI3", "PI4"],          # Purchase Intention
}

# ==========================================================
# BAGIAN C -- MODEL STRUKTURAL (INTI ENGINE BARU)
# ==========================================================
# Bentuk deklaratif berjenjang, mirip JSON bersarang.
#
#   "variabel_endogen": {"prediktor": koefisien_beta, ...}
#
# Konstruk yang TIDAK muncul sebagai kunci di sini otomatis
# diperlakukan sebagai konstruk EKSOGEN (dibangkitkan latent.py
# dengan korelasi bebas antar mereka).
#
# Engine menyortir urutan hitung secara topologis, jadi urutan
# penulisan di bawah TIDAK harus benar. Mediasi berjenjang tetap
# terhitung dengan benar.

STRUCTURAL_MODEL = {
    "PN": {"SI": 0.45},                                  # jalur a1
    "AP": {"SI": 0.40},                                  # jalur a2
    "PI": {"PN": 0.38, "AP": 0.26, "CIA": -0.18, "PE": 0.22},
    #      jalur b1     jalur b2    efek utama moderator  kontrol
}

# ==========================================================
# BAGIAN C2 -- KONSTRUK ORDE DUA
# ==========================================================
# Konstruk orde dua adalah konstruk payung yang tidak punya item
# sendiri, melainkan diukur lewat beberapa DIMENSI orde satu, dan
# tiap dimensi itulah yang punya item.
#
# Bentuk: "NAMA_INDUK": {"dimensi_1": loading, "dimensi_2": loading, ...}
#
# ATURAN:
#   - Nama induk TIDAK boleh ada di CONSTRUCTS (ia tidak punya item).
#   - Tiap dimensi WAJIB ada di CONSTRUCTS lengkap dengan itemnya.
#   - Dimensi tidak boleh dipakai sebagai variabel endogen di
#     STRUCTURAL_MODEL, karena nilainya sudah ditentukan induknya.
#   - Induk BOLEH jadi prediktor maupun variabel endogen biasa.
#
# Loading antar dimensi sebaiknya 0.70 sampai 0.85. Di atas 0.90 dimensi
# jadi nyaris kembar dan validitas diskriminan antar dimensi akan gagal.
SECOND_ORDER = {}

# Contoh pemakaian, salin ke atas kalau dibutuhkan:
# SECOND_ORDER = {"PVal": {"PVal_F": 0.82, "PVal_P": 0.78, "PVal_S": 0.75}}

# ==========================================================
# BAGIAN C3 -- KOVARIAT TERAMATI
# ==========================================================
# Kolom demografi yang boleh dipakai sebagai PREDIKTOR TERAMATI di
# STRUCTURAL_MODEL, biasanya sebagai variabel kontrol. Di AMOS variabel
# ini digambar sebagai KOTAK, bukan lingkaran, karena diukur langsung
# tanpa indikator.
#
# Nilainya dibakukan lebih dulu oleh engine, sehingga koefisien yang
# Anda tulis tetap terbaca sebagai beta terstandardisasi.
#
# SYARAT: nama kolom harus ada di DEMOGRAPHIC_SPEC dan
# GENERATE_DEMOGRAPHICS harus True.
COVARIATES = []

# Contoh: COVARIATES = ["Age", "Income"]
# lalu di STRUCTURAL_MODEL: "PI": {"PN": 0.38, "Age": -0.09, "Income": 0.11}

# ==========================================================
# BAGIAN D3 -- MODERASI OLEH VARIABEL KATEGORIK
# ==========================================================
# Berbeda dari MODERATIONS di Bagian D, yang memakai suku produk
# kontinu untuk moderator berbentuk konstruk laten.
#
# Bentuk ini dipakai kalau moderatornya MEMANG kategorik sejak awal,
# misalnya jenis kelamin, merek kendaraan, atau wilayah. Untuk kasus
# itu, membedakan koefisien per kelompok BUKAN penyederhanaan,
# melainkan justru bentuk yang benar, karena tidak ada gradasi yang
# dibuang. Uji resminya di AMOS adalah Multiple-Group Analysis.
#
# JANGAN memakai bentuk ini untuk moderator kontinu yang dipotong di
# median. Itu membuat proses pembangkit data patah tepat di median
# SAMPEL, padahal median sampel bukan titik yang bermakna secara teori.
# Untuk moderator kontinu, pakai MODERATIONS di Bagian D.
#
# Bentuk: {"outcome", "predictor", "group_column", "deltas"}
# deltas berisi tambahan koefisien per kode kelompok. Rata-rata deltas
# yang dibobot proporsi kelompok sebaiknya mendekati nol supaya
# koefisien utama di STRUCTURAL_MODEL tetap terbaca sebagai efek
# rata-rata seluruh sampel.
GROUP_MODERATIONS = []

# Contoh:
# GROUP_MODERATIONS = [
#     {"outcome": "PI", "predictor": "PN", "group_column": "Gender",
#      "deltas": {1: 0.12, 2: -0.14}},
# ]

# ==========================================================
# BAGIAN D -- MODERASI (SUKU PRODUK KONTINU)
# ==========================================================
# Tiap entri menambahkan satu suku interaksi ke persamaan outcome:
#     outcome = ... + coefficient * (predictor_terpusat x moderator_terpusat)
#
# SYARAT: "moderator" harus juga terdaftar sebagai efek utama di
# STRUCTURAL_MODEL[outcome]. Engine akan menolak jalan kalau tidak,
# karena model interaksi tanpa efek utama moderator tidak dapat
# ditafsirkan.

MODERATIONS = [
    {
        "outcome": "PI",
        "predictor": "PN",
        "moderator": "CIA",
        "coefficient": -0.20,   # negatif = CIA MELEMAHKAN jalur PN -> PI
    },
]

# ==========================================================
# BAGIAN D2 -- KOVARIANS RESIDUAL
# ==========================================================
# Pada mediasi PARALEL, dua mediator biasanya masih berbagi sebab
# yang tidak dimodelkan. Praktik baku di AMOS adalah menggambar
# panah dua arah antar error term kedua mediator. Kalau generator
# tidak menirunya, data yang dihasilkan lebih "bersih" dari model
# yang nanti Anda gambar, dan itu justru sumber salah fit.
#
# Nilai = korelasi antar residual, bukan kovarians mentah.
RESIDUAL_COVARIANCES = [
    {"between": ("PN", "AP"), "correlation": 0.25},
]

# Indikator produk untuk uji interaksi laten di AMOS.
# Strategi "matched_pair": item diurutkan menurut loading, lalu
# dipasangkan satu-satu. Pemusatan rerata ganda (double mean centering).
BUILD_PRODUCT_INDICATORS = True

# ==========================================================
# BAGIAN E -- KORELASI ANTAR KONSTRUK EKSOGEN
# ==========================================================
# Batas atas 0.60, BUKAN 0.70. Alasan historis: pada proyek
# sebelumnya satu pasangan mendarat di 0.699 dan lolos QC internal
# tapi GAGAL validitas diskriminan di AMOS (korelasi laten 0.774).
LATENT_CORR_MIN = 0.30
LATENT_CORR_MAX = 0.60

# ==========================================================
# BAGIAN F -- MODEL PENGUKURAN
# ==========================================================
LOADING_MIN = 0.72
LOADING_MAX = 0.88

# --- KETIDAKSEMPURNAAN STRUKTUR FAKTOR ---
# Tanpa dua parameter di bawah, data yang dihasilkan punya struktur
# faktor bersih sempurna dan CFI-nya menyentuh 0.999. Itu justru
# tanda data rekayasa. Item survei sungguhan selalu sedikit "bocor"
# ke konstruk tetangga, dan sebagian pasangan item berbagi redaksi
# sehingga error-nya berkorelasi.

# Tiga preset terkalibrasi di sandbox pada model 27 indikator, n=400,
# masing-masing diuji lintas enam seed. Hasil terukurnya:
#
#   bersih    CMIN/DF 1.03-1.21  CFI 0.988-0.998  RMSEA 0.009-0.023
#   realistis CMIN/DF 1.28-1.98  CFI 0.947-0.985  RMSEA 0.026-0.050
#   menantang CMIN/DF 1.94-2.63  CFI 0.910-0.953  RMSEA 0.049-0.064
#
# "bersih" sengaja TIDAK dijadikan bawaan. CFI 0.998 pada 27 indikator
# adalah angka yang praktis tidak pernah muncul di survei sungguhan,
# dan justru menjadi tanda paling terang bahwa data direkayasa.
# "menantang" dipakai kalau Anda ingin berlatih membaca Modification
# Indices dan prosedur perbaikan model.
REALISM_PRESETS = {
    "bersih":    {"CROSSLOADING_RATE": 0.20, "CROSSLOADING_SD": 0.09,
                  "CORRELATED_ERROR_RATE": 0.12, "CORRELATED_ERROR_R": 0.15},
    "realistis": {"CROSSLOADING_RATE": 0.45, "CROSSLOADING_SD": 0.14,
                  "CORRELATED_ERROR_RATE": 0.25, "CORRELATED_ERROR_R": 0.24},
    "menantang": {"CROSSLOADING_RATE": 0.60, "CROSSLOADING_SD": 0.19,
                  "CORRELATED_ERROR_RATE": 0.35, "CORRELATED_ERROR_R": 0.32},
}
REALISM_PRESET = "realistis"

CROSSLOADING_RATE = REALISM_PRESETS[REALISM_PRESET]["CROSSLOADING_RATE"]
CROSSLOADING_SD = REALISM_PRESETS[REALISM_PRESET]["CROSSLOADING_SD"]
CORRELATED_ERROR_RATE = REALISM_PRESETS[REALISM_PRESET]["CORRELATED_ERROR_RATE"]
CORRELATED_ERROR_R = REALISM_PRESETS[REALISM_PRESET]["CORRELATED_ERROR_R"]

# ==========================================================
# BAGIAN G -- SKALA LIKERT DAN DISTRIBUSI
# ==========================================================
# Proporsi kategori 1..k, total harus 1.0. Hasil pencarian numerik
# (lihat ordinal_engine.find_proportions), bukan tebakan.
# Dihasilkan ordinal_engine.find_proportions() lewat diskretisasi
# normal dua parameter, bukan tebakan. Bentuknya dijamin tunggal
# puncak. Nilai di bawah sudah diverifikasi ulang di sandbox.
LIKERT_PROPORTIONS_BY_SCALE = {
    4: (0.0225, 0.1918, 0.4490, 0.3367),           # mean 3.10, sd 0.78
    5: (0.0089, 0.0696, 0.2443, 0.3672, 0.3100),   # mean 3.90, sd 0.95
    6: (0.0038, 0.0302, 0.1284, 0.2803, 0.3142, 0.2430),  # mean 4.60, sd 1.10
    7: (0.0044, 0.0207, 0.0721, 0.1646, 0.2469, 0.2435, 0.2478),  # mean 5.35, sd 1.35
}

# Pergeseran kecil ambang batas antar item, supaya tiap indikator
# punya "tingkat kesulitan" sedikit berbeda.
THRESHOLD_JITTER_SD = 0.10

# ==========================================================
# BAGIAN H -- REALISME JAWABAN MANUSIA
# ==========================================================
# Tiap tombol di bawah menambah realisme TAPI punya biaya terhadap
# indeks kecocokan. Nilai default sudah diuji aman.

# Heteroskedastisitas error per responden. Sebagian orang menjawab
# sangat konsisten dengan sikap dasarnya (error kecil), sebagian lain
# lebih berisik. Kalau semua responden diberi sigma error identik,
# variasi individual ini hilang. Sigma dasar tiap item digoyang dalam
# rentang (1 - nilai) sampai (1 + nilai) khusus per responden.
# Teknik ini diwarisi dari generator versi sebelumnya, di mana
# rentangnya 0,90 sampai 1,10 yang setara nilai 0.10.
ERROR_SIGMA_JITTER = 0.10

# Kecenderungan responden menjawab setuju terlepas isi pertanyaan.
# Ini menciptakan faktor metode bersama. Biaya: menaikkan varians
# faktor tunggal di uji Harman. Aman sampai sekitar 0.15.
ACQUIESCENCE_SD = 0.10

# Perbedaan kecenderungan memakai ujung skala antar responden.
# Biaya: hampir nol. Aman sampai sekitar 0.18.
EXTREME_STYLE_SD = 0.12

# Proporsi responden yang menjawab nyaris lurus satu angka.
# Biaya: LANGSUNG menurunkan CFI dan menaikkan RMSEA. Naikkan hanya
# kalau memang ingin melatih prosedur pembersihan data.
STRAIGHTLINER_RATE = 0.015

# Proporsi sel yang dikosongkan. AMOS bisa menangani lewat FIML,
# tapi banyak prosedur lain tidak. Default nol.
MISSING_RATE = 0.0

# ==========================================================
# BAGIAN I -- DEMOGRAFI
# ==========================================================
GENERATE_DEMOGRAPHICS = True

DEMOGRAPHIC_SPEC = {
    "Gender":    {"labels": ["Laki-laki", "Perempuan"],
                  "probs": [0.54, 0.46], "ses_loading": 0.0},
    "Age":       {"labels": ["17-25", "26-35", "36-45", "46-55", "56-64"],
                  "probs": [0.28, 0.34, 0.22, 0.11, 0.05], "ses_loading": 0.25},
    "Education": {"labels": ["SMA/SMK", "Diploma", "Sarjana", "Pascasarjana"],
                  "probs": [0.31, 0.19, 0.41, 0.09], "ses_loading": 0.55},
    "Income":    {"labels": ["<3jt", "3-6jt", "6-10jt", ">10jt"],
                  "probs": [0.27, 0.38, 0.24, 0.11], "ses_loading": 0.70},
    "VehicleOwn": {"labels": ["Motor konvensional", "Mobil konvensional",
                              "Keduanya", "Belum punya"],
                   "probs": [0.49, 0.14, 0.28, 0.09], "ses_loading": 0.45},
}

# ==========================================================
# BAGIAN J -- AMBANG KELAYAKAN (dipakai quality_control.py)
# ==========================================================
TARGET_MEAN = (3.70, 4.15)      # dalam satuan skala asli
TARGET_STD = (0.80, 1.10)
MAX_DUPLICATE_ROWS = 0

CRONBACH_MIN, CRONBACH_MAX = 0.70, 0.93
CR_MIN = 0.70
AVE_MIN = 0.50

# Margin optimisme lapis satu. Estimasi komponen utama BIAS KE ATAS
# terhadap loading CFA sungguhan. Diukur ulang di sandbox pada model ini,
# selisih AVE rata-rata +0.083 dengan rentang +0.072 sampai +0.098.
# Angka ini konsisten dengan selisih +0.08 yang tercatat pada proyek
# sebelumnya, jadi biasnya sifat sistematis metode.
#
# PERHATIAN, margin ini dipasang di angka RATA-RATA bias, bukan di angka
# terburuk. Artinya sebagian konstruk tetap punya bias melampaui margin.
# Itu DISENGAJA dan aman, karena sejak lapis dua dipindah ke DALAM loop
# optimizer, gerbang kelayakan yang sesungguhnya dijaga lapis dua.
# Margin ini kini berfungsi sebagai penghemat waktu, yaitu menyaring
# kandidat yang jelas akan gagal supaya CFA tidak perlu dijalankan,
# BUKAN sebagai pengaman terakhir. Kalau suatu saat FIT_CHECK_IN_OPTIMIZER
# dimatikan, margin ini harus dinaikkan ke 0.10 karena perannya berubah
# menjadi pengaman.
PCA_OPTIMISM_MARGIN = 0.08

# Margin aman validitas diskriminan. Rumus disattenuation sendiri
# masih sedikit meremehkan angka CFA sungguhan (selisih terukur
# sampai 0.036 pada kasus nyata), jadi margin ini bukan hiasan.
DISCRIMINANT_SAFETY_MARGIN = 0.05

# Batas atas R2 yang masih terasa wajar untuk data perilaku.
R2_MAX = 0.80

MAX_OPTIMIZER_ITERATIONS = 60

# Ambang indeks kecocokan yang diperiksa optimizer pada LAPIS DUA.
# Satu siklus penuh termasuk CFA memakan sekitar 0,15 detik pada model
# 27 indikator n=400, jadi lapis dua muat di dalam loop dan tidak perlu
# dijalankan manual belakangan.
FIT_CHECK_IN_OPTIMIZER = True
FIT_THRESHOLDS = {
    "CMIN/DF_max": 2.00,
    "CFI_min": 0.950,
    "TLI_min": 0.950,
    "RMSEA_max": 0.070,
}

# ==========================================================
# BAGIAN K -- KELUARAN
# ==========================================================
OUTPUT_DIR = "output"
OUTPUT_FILENAME = "output/dataset_sem.csv"


# ==========================================================
# TURUNAN OTOMATIS -- jangan diubah manual
# ==========================================================
def all_constructs():
    return list(CONSTRUCTS.keys())


def second_order_parents():
    """Konstruk payung. Tidak punya item, tidak muncul di CONSTRUCTS."""
    return list(SECOND_ORDER.keys())


def second_order_dimensions():
    """Dimensi orde satu. Punya item, tapi nilainya ditentukan induknya."""
    return [d for dims in SECOND_ORDER.values() for d in dims]


def latent_variables():
    """Semua variabel laten, termasuk induk orde dua yang tanpa item."""
    return list(CONSTRUCTS.keys()) + second_order_parents()


def effective_structural_model():
    """
    STRUCTURAL_MODEL yang sudah diperluas dengan jalur induk ke dimensi.

    Dengan memperlakukan tiap dimensi sebagai variabel endogen yang
    prediktor tunggalnya adalah induknya, seluruh mesin yang sudah ada
    (penyortiran topologis dan penskalaan residual) langsung berlaku
    tanpa cabang logika baru.
    """
    model = {o: dict(p) for o, p in STRUCTURAL_MODEL.items()}
    for induk, dimensi in SECOND_ORDER.items():
        for dim, loading in dimensi.items():
            model[dim] = {induk: float(loading)}
    return model


def endogenous_constructs():
    return list(effective_structural_model().keys())


def exogenous_constructs():
    endo = set(effective_structural_model())
    return [c for c in latent_variables() if c not in endo]


def all_items():
    items = []
    for daftar in CONSTRUCTS.values():
        items.extend(daftar)
    return items


def likert_proportions():
    if SCALE_POINTS not in LIKERT_PROPORTIONS_BY_SCALE:
        raise ValueError(
            f"SCALE_POINTS={SCALE_POINTS} belum punya proporsi terkalibrasi. "
            "Jalankan ordinal_engine.find_proportions() lebih dulu."
        )
    return LIKERT_PROPORTIONS_BY_SCALE[SCALE_POINTS]


if __name__ == "__main__":
    print("=== CEK CONFIG.PY ===")
    print("Konstruk eksogen :", exogenous_constructs())
    print("Konstruk endogen :", endogenous_constructs())
    print("Induk orde dua   :", second_order_parents() or "tidak ada")
    print("Kovariat teramati:", COVARIATES or "tidak ada")
    print("Moderasi kategorik:", len(GROUP_MODERATIONS))
    print("Jumlah indikator :", len(all_items()))
    print("Jumlah responden :", N_RESPONDENTS)
    print("Titik skala      :", SCALE_POINTS)
    print("Proporsi kategori:", likert_proportions(),
          "total =", round(sum(likert_proportions()), 4))
    print("Suku moderasi    :", len(MODERATIONS))
