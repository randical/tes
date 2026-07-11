"""
structural.py
--------------
Menambahkan lapisan MODEL STRUKTURAL di atas skor laten eksogen dari
latent.py: CL (config.STRUCTURAL_OUTCOME) dihitung dari 6 konstruk
UTAUT2 (H1-H6) lewat jalur berbobot -- TAPI kekuatan jalur itu
berbeda tergantung level EC tiap responden (H7a-H7f: EC sebagai
MODERATOR, bukan prediktor langsung ke CL).
"""

import numpy as np

import config


class StructuralModel:
    """
    Menghitung skor laten CL dari 6 konstruk prediktor, dengan
    koefisien jalur yang berbeda untuk responden ber-EC rendah
    (STRUCTURAL_PATHS_LOW_EC) vs ber-EC tinggi (STRUCTURAL_PATHS_HIGH_EC).
    """

    def __init__(self, seed=None):
        seed = seed if seed is not None else config.RANDOM_SEED + 1
        np.random.seed(seed)
        self.outcome = config.STRUCTURAL_OUTCOME
        self.moderator = config.MODERATOR_CONSTRUCT
        self.predictors = config.STRUCTURAL_PREDICTORS
        self.paths_low = config.STRUCTURAL_PATHS_LOW_EC
        self.paths_high = config.STRUCTURAL_PATHS_HIGH_EC

    def apply(self, df_latent_exogenous):
        """
        KENAPA:
        Moderasi artinya kekuatan pengaruh PE..PV terhadap CL BERBEDA
        tergantung level EC responden -- bukan EC ikut menjumlah
        langsung ke CL seperti prediktor biasa. Karena itu, tiap
        responden perlu "dipilihkan" set koefisien jalur yang sesuai
        levelnya: responden EC rendah pakai STRUCTURAL_PATHS_LOW_EC,
        EC tinggi pakai STRUCTURAL_PATHS_HIGH_EC.

        BAGAIMANA:
        1) Bagi responden jadi dua kelompok berdasarkan median skor EC.
        2) Untuk tiap responden, ambil baris koefisien jalur sesuai
           kelompoknya (pakai np.where untuk memilih baris per
           responden -- bukan satu vektor b yang sama untuk semua
           orang seperti versi sebelumnya).
        3) Kalikan skor prediktor dengan koefisien jalur MASING-MASING
           responden, lalu jumlahkan per baris (bukan @ seperti
           sebelumnya, karena sekarang koefisiennya tidak seragam
           antar responden).
        4) Tambahkan residu, lalu standarkan ulang seperti biasa.
        """
        moderator_score = df_latent_exogenous[self.moderator]
        median_ec = moderator_score.median()
        is_high = (moderator_score >= median_ec).values  # (200,) True/False

        b_low = np.array([self.paths_low[p] for p in self.predictors])
        b_high = np.array([self.paths_high[p] for p in self.predictors])

        # np.where(kondisi, nilai_jika_true, nilai_jika_false), tapi di sini
        # kondisinya per-baris (is_high[:, None] membuat bentuknya cocok
        # untuk "disiarkan" ke semua kolom) -- hasilnya: tiap baris punya
        # koefisien jalurnya sendiri, tergantung is_high baris itu.
        b_per_respondent = np.where(is_high[:, None], b_high, b_low)  # (200, 6)

        predictor_scores = df_latent_exogenous[self.predictors].values  # (200, 6)
        structural_part = (predictor_scores * b_per_respondent).sum(axis=1)  # (200,)

        explained_variance = min(structural_part.var(), 0.95)
        residual_variance = max(1 - explained_variance, 0.05)
        zeta = np.random.normal(0, np.sqrt(residual_variance), len(df_latent_exogenous))

        outcome_score = structural_part + zeta
        outcome_score = (outcome_score - outcome_score.mean()) / outcome_score.std()

        self.r_squared = explained_variance
        self.n_high = int(is_high.sum())
        self.n_low = int((~is_high).sum())

        df_result = df_latent_exogenous.copy()
        df_result[self.outcome] = outcome_score
        return df_result


# ==========================================================
# BLOK PENGECEKAN MANDIRI
# ==========================================================
if __name__ == "__main__":
    from latent import LatentGenerator

    latent_gen = LatentGenerator()
    df_exogenous = latent_gen.sample_latent()

    structural = StructuralModel()
    df_latent = structural.apply(df_exogenous)

    print("=== CEK STRUCTURAL.PY (versi moderasi) ===")
    print(f"Jumlah responden Low EC : {structural.n_low}")
    print(f"Jumlah responden High EC: {structural.n_high}")
    print(f"R^2 teoretis (gabungan) : {structural.r_squared:.3f}")

    median_ec = df_latent[config.MODERATOR_CONSTRUCT].median()
    low_group = df_latent[df_latent[config.MODERATOR_CONSTRUCT] < median_ec]
    high_group = df_latent[df_latent[config.MODERATOR_CONSTRUCT] >= median_ec]

    print("\nKorelasi tiap prediktor dengan CL, per kelompok "
          "(harusnya mengikuti pola H7a-H7f di config.py):")
    print(f"{'Prediktor':<10}{'Low EC':>10}{'High EC':>10}  Sesuai hipotesis?")
    for p in config.STRUCTURAL_PREDICTORS:
        corr_low = low_group[[p, "CL"]].corr().iloc[0, 1]
        corr_high = high_group[[p, "CL"]].corr().iloc[0, 1]
        arah_diharapkan = "High>Low" if config.STRUCTURAL_PATHS_HIGH_EC[p] > config.STRUCTURAL_PATHS_LOW_EC[p] else "High<Low"
        arah_aktual = "High>Low" if corr_high > corr_low else "High<Low"
        cocok = "OK" if arah_diharapkan == arah_aktual else "beda arah"
        print(f"{p:<10}{corr_low:>10.3f}{corr_high:>10.3f}  {cocok}")
