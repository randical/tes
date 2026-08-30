"""
config.py -- PROYEK EV KALIMANTAN BARAT
=======================================
TAM Analysis, Perceived Risk, Government Promotion terhadap Adoption
Intention Kendaraan Listrik melalui Demografi sebagai Variabel Moderating
di Kalimantan Barat.

PERINGATAN. Dataset yang dihasilkan berkas ini SINTETIS. Angkanya tidak
berasal dari responden mana pun. Lihat MODUL_0 bagian 2.
"""

# ==========================================================
# BAGIAN A -- PENGATURAN UMUM
# ==========================================================
RANDOM_SEED = 52500          # dipilih lewat sweep.py, lihat laporan bagian 8
N_RESPONDENTS = 200          # dikunci oleh peneliti
SCALE_POINTS = 5             # sesuai kuesioner Bagian B

# ==========================================================
# BAGIAN B -- KONSTRUK DAN INDIKATOR (28 item, sesuai kuesioner)
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
# Empat prediktor sejajar, persis seperti hipotesis awal peneliti.
# TIDAK ada jalur PEOU ke PU, karena peneliti memutuskan delapan
# hipotesis awal tidak berubah.
STRUCTURAL_MODEL = {
    "AI": {"PEOU": 0.17, "PU": 0.32, "PR": -0.22, "GP": 0.20},
}

SECOND_ORDER = {}
COVARIATES = []

# ==========================================================
# BAGIAN D3 -- MODERASI KATEGORIK (H5 sampai H8)
# ==========================================================
# Moderator resmi Gender. Kode 1 Laki-laki, kode 2 Perempuan.
# Uji resmi di AMOS adalah Multiple-Group Analysis.
GROUP_MODERATIONS = [
    {"outcome": "AI", "predictor": "PEOU", "group_column": "Gender",
     "deltas": {1: -0.16, 2: 0.17}},
    {"outcome": "AI", "predictor": "PU", "group_column": "Gender",
     "deltas": {1: 0.03, 2: -0.03}},
    {"outcome": "AI", "predictor": "PR", "group_column": "Gender",
     "deltas": {1: 0.15, 2: -0.16}},
    {"outcome": "AI", "predictor": "GP", "group_column": "Gender",
     "deltas": {1: 0.02, 2: -0.02}},
]

MODERATIONS = []
RESIDUAL_COVARIANCES = []
BUILD_PRODUCT_INDICATORS = False

# ==========================================================
# BAGIAN E -- KORELASI ANTAR KONSTRUK EKSOGEN
# ==========================================================
LATENT_CORR_MIN = 0.30
LATENT_CORR_MAX = 0.60

LATENT_CORR_OVERRIDE = {
    ("PEOU", "PU"): 0.52,
    ("PEOU", "PR"): -0.24,
    ("PEOU", "GP"): 0.33,
    ("PU", "PR"): -0.20,
    ("PU", "GP"): 0.40,
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
OUTPUT_FILENAME = "output/dataset_EV_Kalbar_SINTETIS.csv"


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
    print("=== CEK CONFIG.PY -- PROYEK EV KALBAR ===")
    print("Konstruk eksogen  :", exogenous_constructs())
    print("Konstruk endogen  :", endogenous_constructs())
    print("Jumlah indikator  :", len(all_items()))
    print("Jumlah responden  :", N_RESPONDENTS)
    print("Moderasi kategorik:", len(GROUP_MODERATIONS))

    urut = ["PEOU", "PU", "PR", "GP"]
    b = np.array([STRUCTURAL_MODEL["AI"][k] for k in urut])
    R = np.eye(4)
    for i in range(4):
        for j in range(i + 1, 4):
            r = LATENT_CORR_OVERRIDE.get((urut[i], urut[j]),
                                         LATENT_CORR_OVERRIDE.get((urut[j], urut[i])))
            R[i, j] = R[j, i] = r
    print("\nR2 rancangan AI tanpa suku moderasi:", round(float(b @ R @ b), 4))
    print("Eigenvalue matriks korelasi eksogen:",
          np.round(np.linalg.eigvalsh(R), 4))

    for g in GROUP_MODERATIONS:
        p = g["predictor"]
        base = STRUCTURAL_MODEL["AI"][p]
        w = DEMOGRAPHIC_SPEC["Gender"]["probs"]
        rata = w[0] * g["deltas"][1] + w[1] * g["deltas"][2]
        print(f"  {p:<5} L={base + g['deltas'][1]:+.3f}  "
              f"P={base + g['deltas'][2]:+.3f}  "
              f"selisih={g['deltas'][2] - g['deltas'][1]:+.3f}  "
              f"rata-rata delta={rata:+.4f}")
