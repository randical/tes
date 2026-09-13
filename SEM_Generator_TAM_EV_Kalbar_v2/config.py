"""
config.py -- PROYEK EV KALIMANTAN BARAT, MODEL REVISI
=====================================================
TAM Analysis, Perceived Risk, Government Promotion terhadap Adoption
Intention Kendaraan Listrik dengan Jenis Kelamin sebagai Variabel
Moderating di Kalimantan Barat.

PERUBAHAN DARI MODEL LAMA
  1. Ditambahkan jalur PEOU ke PU, sesuai TAM Davis (1989).
     Akibatnya PU berubah status dari eksogen menjadi ENDOGEN.
  2. Moderator dikunci ke Jenis Kelamin, bukan "Demografi".
  3. Jumlah hipotesis bertambah dari 8 menjadi 9.

PERINGATAN. Dataset yang dihasilkan berkas ini SINTETIS. Angkanya tidak
berasal dari responden mana pun. Lihat MODUL_0 bagian 2.
"""

# ==========================================================
# BAGIAN A -- PENGATURAN UMUM
# ==========================================================
RANDOM_SEED = 114000         # dipilih lewat sweep.py, lihat laporan bagian 8
N_RESPONDENTS = 200          # dikunci oleh peneliti, alasan anggaran
SCALE_POINTS = 5

# ==========================================================
# BAGIAN B -- KONSTRUK DAN INDIKATOR (28 item, kuesioner tidak berubah)
# ==========================================================
CONSTRUCTS = {
    "PEOU": ["PEOU1", "PEOU2", "PEOU3", "PEOU4", "PEOU5"],
    "PU":   ["PU1", "PU2", "PU3", "PU4", "PU5"],
    "PR":   ["PR1", "PR2", "PR3", "PR4", "PR5", "PR6", "PR7"],
    "GP":   ["GP1", "GP2", "GP3", "GP4", "GP5"],
    "AI":   ["AI1", "AI2", "AI3", "AI4", "AI5", "AI6"],
}

# ==========================================================
# BAGIAN C -- MODEL STRUKTURAL
# ==========================================================
# H1 PEOU -> PU    positif KUAT, ini fondasi TAM
# H2 PEOU -> AI    positif LEMAH, sengaja kecil, lihat catatan di bawah
# H3 PU   -> AI    positif, jalur langsung terkuat
# H4 PR   -> AI    negatif
# H5 GP   -> AI    positif
#
# Catatan soal H2. Davis (1989) sendiri menyatakan kemudahan penggunaan
# kemungkinan merupakan anteseden manfaat, bukan penentu langsung niat.
# Jalur langsungnya dibuat kecil supaya pola hasil sesuai teori. Pengaruh
# PEOU yang sesungguhnya muncul sebagai efek TIDAK LANGSUNG lewat PU,
# yaitu 0.52 x 0.34 = 0.177, yang justru lebih besar dari jalur
# langsungnya sendiri.
STRUCTURAL_MODEL = {
    "PU": {"PEOU": 0.52},
    "AI": {"PEOU": 0.12, "PU": 0.34, "PR": -0.22, "GP": 0.20},
}

SECOND_ORDER = {}
COVARIATES = []

# ==========================================================
# BAGIAN D3 -- MODERASI KATEGORIK (H6 sampai H9)
# ==========================================================
# Moderator Jenis Kelamin. Kode 1 Laki-laki, kode 2 Perempuan.
# Uji resmi di AMOS adalah Multiple-Group Analysis dengan Delta chi-square.
#
# Rata-rata delta dibobot proporsi kelompok dibuat mendekati nol, supaya
# koefisien di STRUCTURAL_MODEL tetap terbaca sebagai efek rata-rata.
#
# H6 dan H8 dirancang TERDUKUNG, dasarnya Venkatesh dan Morris (2000)
# untuk kemudahan penggunaan, serta Byrnes dkk. (1999) untuk risiko.
# H7 dan H9 dirancang TIDAK TERDUKUNG, karena tidak ada dasar teori yang
# kuat bahwa manfaat ekonomi maupun daya tarik subsidi berbeda menurut
# jenis kelamin.
GROUP_MODERATIONS = [
    {"outcome": "AI", "predictor": "PEOU", "group_column": "Gender",
     "deltas": {1: -0.15, 2: 0.156}},        # L -0.03  vs  P 0.276
    {"outcome": "AI", "predictor": "PU", "group_column": "Gender",
     "deltas": {1: 0.03, 2: -0.031}},        # L  0.37  vs  P 0.309
    {"outcome": "AI", "predictor": "PR", "group_column": "Gender",
     "deltas": {1: 0.15, 2: -0.156}},        # L -0.07  vs  P -0.376
    {"outcome": "AI", "predictor": "GP", "group_column": "Gender",
     "deltas": {1: 0.02, 2: -0.021}},        # L  0.22  vs  P 0.179
]

MODERATIONS = []
RESIDUAL_COVARIANCES = []
BUILD_PRODUCT_INDICATORS = False

# ==========================================================
# BAGIAN E -- KORELASI ANTAR KONSTRUK EKSOGEN
# ==========================================================
# PENTING. Setelah jalur PEOU ke PU ditambahkan, PU menjadi ENDOGEN,
# sehingga hanya tersisa TIGA konstruk eksogen, yaitu PEOU, PR, dan GP.
# Korelasi PU dengan PR maupun GP tidak lagi ditetapkan di sini,
# melainkan TERBENTUK SENDIRI lewat PEOU.
LATENT_CORR_MIN = 0.30
LATENT_CORR_MAX = 0.60

LATENT_CORR_OVERRIDE = {
    ("PEOU", "PR"): -0.24,
    ("PEOU", "GP"): 0.33,
    ("PR", "GP"): -0.15,
}

# ==========================================================
# BAGIAN F -- MODEL PENGUKURAN
# ==========================================================
LOADING_MIN = 0.71
LOADING_MAX = 0.88

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

# PR1 dan PR2 sama-sama risiko finansial, PR3 dan PR4 sama-sama risiko
# kinerja. Pasangan seperti itu di data sungguhan hampir selalu punya
# error berkorelasi.
FORCED_ERROR_PAIRS = [("PR1", "PR2"), ("PR3", "PR4")]

# ==========================================================
# BAGIAN G -- SKALA LIKERT DAN DISTRIBUSI
# ==========================================================
LIKERT_PROPORTIONS_BY_SCALE = {
    4: (0.0225, 0.1918, 0.4490, 0.3367),
    5: (0.0089, 0.0696, 0.2443, 0.3672, 0.3100),
    6: (0.0038, 0.0302, 0.1284, 0.2803, 0.3142, 0.2430),
    7: (0.0044, 0.0207, 0.0721, 0.1646, 0.2469, 0.2435, 0.2478),
}

# Terjemahan konteks Kalimantan Barat ke angka.
#   PU  tertinggi   argumen hemat bahan bakar paling mudah diterima
#   PEOU tinggi     mengoperasikan EV dianggap tidak sulit
#   PR  cukup tinggi SPKLU terbatas, harga dan baterai jadi kekhawatiran
#   AI  sedang      niat ada, tertahan hambatan praktis
#   GP  TERENDAH    promosi dan subsidi kurang terasa di luar Jawa
CONSTRUCT_LIKERT_TARGET = {
    "PEOU": (3.85, 0.92),
    "PU":   (4.05, 0.88),
    "PR":   (3.78, 0.96),
    "GP":   (3.12, 1.05),
    "AI":   (3.68, 0.98),
}
CONSTRUCT_MEAN_TOLERANCE = 0.18

THRESHOLD_JITTER_SD = 0.10

# ==========================================================
# BAGIAN H -- REALISME JAWABAN MANUSIA
# ==========================================================
ERROR_SIGMA_JITTER = 0.10
ACQUIESCENCE_SD = 0.10
EXTREME_STYLE_SD = 0.12
STRAIGHTLINER_RATE = 0.015
MISSING_RATE = 0.0

# ==========================================================
# BAGIAN I -- DEMOGRAFI (mengikuti Bagian A kuesioner)
# ==========================================================
GENERATE_DEMOGRAPHICS = True

# CATATAN. Seluruh proporsi di bawah adalah ASUMSI RANCANGAN yang meniru
# survei daring non-probabilitas di koridor perkotaan Kalbar. Ini BUKAN
# angka BPS dan tidak boleh dikutip sebagai statistik daerah.
DEMOGRAPHIC_SPEC = {
    "Domisili": {
        "labels": ["Kabupaten Kapuas Hulu", "Kabupaten Melawi",
                   "Kabupaten Sekadau", "Kabupaten Kayong Utara",
                   "Kabupaten Bengkayang", "Kabupaten Sintang",
                   "Kabupaten Ketapang", "Kabupaten Landak",
                   "Kabupaten Sanggau", "Kabupaten Sambas",
                   "Kabupaten Mempawah", "Kota Singkawang",
                   "Kabupaten Kubu Raya", "Kota Pontianak"],
        "probs": [0.01, 0.01, 0.01, 0.01, 0.02, 0.03, 0.04, 0.04,
                  0.05, 0.07, 0.08, 0.11, 0.18, 0.34],
        "ses_loading": 0.20},

    "Gender": {
        "labels": ["Laki-laki", "Perempuan"],
        "probs": [0.51, 0.49],
        "ses_loading": 0.0},

    "Usia": {
        "labels": ["< 20 tahun", "21-30 tahun", "31-40 tahun",
                   "41-50 tahun", "> 50 tahun"],
        "probs": [0.08, 0.42, 0.29, 0.15, 0.06],
        "ses_loading": 0.28},

    "Pendidikan": {
        "labels": ["SMA/SMK", "D3", "S1", "S2", "S3"],
        "probs": [0.38, 0.12, 0.42, 0.07, 0.01],
        "ses_loading": 0.55},

    "Pekerjaan": {
        "labels": ["Pelajar/Mahasiswa", "Ibu Rumah Tangga", "Lainnya",
                   "Wiraswasta", "Pegawai Swasta", "ASN/TNI/POLRI",
                   "BUMN/BUMD"],
        "probs": [0.20, 0.07, 0.03, 0.19, 0.28, 0.17, 0.06],
        "ses_loading": 0.30},

    "Pendapatan": {
        "labels": ["< Rp 3.000.000", "Rp 3.000.000 - Rp 5.000.000",
                   "Rp 5.000.001 - Rp 10.000.000",
                   "Rp 10.000.001 - Rp 15.000.000", "> Rp 15.000.000"],
        "probs": [0.31, 0.33, 0.24, 0.08, 0.04],
        "ses_loading": 0.72},

    "Kendaraan": {
        "labels": ["Tidak memiliki kendaraan", "Sepeda motor", "Mobil",
                   "Sepeda motor dan mobil"],
        "probs": [0.07, 0.59, 0.06, 0.28],
        "ses_loading": 0.45},

    "PernahPakaiEV": {
        "labels": ["Tidak", "Ya"],
        "probs": [0.77, 0.23],
        "ses_loading": 0.18,
        "construct_loading": {"PEOU": 0.26}},
}

# ==========================================================
# BAGIAN J -- AMBANG KELAYAKAN
# ==========================================================
TARGET_MEAN = (3.40, 4.00)
TARGET_STD = (0.80, 1.15)
MAX_DUPLICATE_ROWS = 0

CRONBACH_MIN, CRONBACH_MAX = 0.70, 0.93
CR_MIN = 0.70
AVE_MIN = 0.50

PCA_OPTIMISM_MARGIN = 0.08
DISCRIMINANT_SAFETY_MARGIN = 0.05
R2_MAX = 0.80

MAX_OPTIMIZER_ITERATIONS = 60

FIT_CHECK_IN_OPTIMIZER = True
FIT_THRESHOLDS = {
    "CMIN/DF_max": 1.60,
    "CFI_min": 0.950,
    "TLI_min": 0.945,
    "RMSEA_max": 0.055,
}

# ==========================================================
# BAGIAN K -- KELUARAN
# ==========================================================
OUTPUT_DIR = "output"
OUTPUT_FILENAME = "output/dataset_EV_Kalbar_REVISI_SINTETIS.csv"


# ==========================================================
# TURUNAN OTOMATIS
# ==========================================================
def all_constructs():
    return list(CONSTRUCTS.keys())


def second_order_parents():
    return list(SECOND_ORDER.keys())


def second_order_dimensions():
    return [d for dims in SECOND_ORDER.values() for d in dims]


def latent_variables():
    return list(CONSTRUCTS.keys()) + second_order_parents()


def effective_structural_model():
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
        raise ValueError(f"SCALE_POINTS={SCALE_POINTS} belum terkalibrasi.")
    return LIKERT_PROPORTIONS_BY_SCALE[SCALE_POINTS]


if __name__ == "__main__":
    import numpy as np
    print("=== CEK CONFIG.PY -- MODEL REVISI EV KALBAR ===")
    print("Konstruk eksogen  :", exogenous_constructs())
    print("Konstruk endogen  :", endogenous_constructs())
    print("Jumlah indikator  :", len(all_items()))
    print("Jumlah responden  :", N_RESPONDENTS)
    print("Moderasi kategorik:", len(GROUP_MODERATIONS))

    b_pu = STRUCTURAL_MODEL["PU"]["PEOU"]
    print(f"\nR2 rancangan PU: {b_pu**2:.4f}")

    # korelasi implied antar keempat prediktor AI
    urut = ["PEOU", "PU", "PR", "GP"]
    R = np.eye(4)
    def eks(a, b):
        return LATENT_CORR_OVERRIDE.get((a, b),
               LATENT_CORR_OVERRIDE.get((b, a)))
    R[0, 1] = R[1, 0] = b_pu                       # PEOU-PU
    R[0, 2] = R[2, 0] = eks("PEOU", "PR")
    R[0, 3] = R[3, 0] = eks("PEOU", "GP")
    R[1, 2] = R[2, 1] = b_pu * eks("PEOU", "PR")   # induksi lewat PEOU
    R[1, 3] = R[3, 1] = b_pu * eks("PEOU", "GP")
    R[2, 3] = R[3, 2] = eks("PR", "GP")
    b = np.array([STRUCTURAL_MODEL["AI"][k] for k in urut])
    print(f"R2 rancangan AI tanpa suku moderasi: {float(b @ R @ b):.4f}")
    print("Eigenvalue matriks korelasi prediktor:", np.round(np.linalg.eigvalsh(R), 4))

    tak_langsung = b_pu * STRUCTURAL_MODEL["AI"]["PU"]
    langsung = STRUCTURAL_MODEL["AI"]["PEOU"]
    print(f"\nPEOU ke AI, langsung        : {langsung:.4f}")
    print(f"PEOU ke AI, tidak langsung  : {tak_langsung:.4f}")
    print(f"PEOU ke AI, total           : {langsung + tak_langsung:.4f}")
    print("Efek tidak langsung LEBIH BESAR dari langsung, sesuai TAM.")

    print("\nRancangan moderasi jenis kelamin")
    w = DEMOGRAPHIC_SPEC["Gender"]["probs"]
    for g in GROUP_MODERATIONS:
        p_ = g["predictor"]
        base = STRUCTURAL_MODEL["AI"][p_]
        rata = w[0] * g["deltas"][1] + w[1] * g["deltas"][2]
        print(f"  {p_:<5} L={base + g['deltas'][1]:+.3f}  "
              f"P={base + g['deltas'][2]:+.3f}  "
              f"selisih={g['deltas'][2] - g['deltas'][1]:+.3f}  "
              f"rata-rata delta={rata:+.4f}")
