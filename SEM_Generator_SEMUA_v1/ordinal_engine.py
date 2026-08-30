"""
ordinal_engine.py
------------------
Mengubah skor kontinu (hasil indicator.py) menjadi skala Likert 1-5,
memakai pendekatan ambang batas (threshold) pada skor laten kontinu --
bukan sekadar pembulatan.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

import config


class OrdinalEngine:
    """
    Membungkus langkah-langkah untuk mengubah kolom angka kontinu
    menjadi kategori Likert 1-5, dengan sedikit variasi antar-item
    supaya distribusinya tidak identik semua.
    """

    def __init__(self, seed=None):
        seed = seed if seed is not None else config.RANDOM_SEED + 2
        np.random.seed(seed)
        self.base_thresholds = self._build_base_thresholds()

    def _build_base_thresholds(self):
        """
        KENAPA:
        Kita ingin distribusi akhir condong ke kanan (banyak jawaban
        4-5) sesuai config.LIKERT_PROPORTIONS, bukan asal potong rata
        di titik -1, 0, 1. Cara paling tepat adalah menghitung batas
        z-score dari proporsi yang kita inginkan, bukan sebaliknya.

        BAGAIMANA:
        Proporsi [0.05, 0.10, 0.20, 0.30, 0.35] diubah jadi kumulatif
        [0.05, 0.15, 0.35, 0.65, 1.00], lalu norm.ppf() mengubah tiap
        angka kumulatif itu jadi batas z-score. Angka kumulatif 1.00
        (100%) tidak perlu dijadikan batas -- makanya diambil semua
        kecuali yang terakhir dengan [:-1].
        """
        proportions = config.LIKERT_PROPORTIONS
        cumulative = np.cumsum(proportions)
        thresholds = norm.ppf(cumulative[:-1])
        return thresholds

    def convert_one_column(self, scores):
        """
        KENAPA:
        Item survei tidak semuanya "sama sulitnya" -- PE1 mungkin
        sedikit lebih gampang disetujui dibanding PE3. Kalau semua
        item memakai threshold dasar yang identik, distribusi tiap
        indikator akan terasa seragam dan tidak natural. Karena itu
        tiap kolom diberi sedikit pergeseran (jitter) dari threshold
        dasarnya.

        BAGAIMANA:
        Tambahkan angka acak kecil ke threshold dasar, urutkan lagi
        (wajib menaik supaya searchsorted bekerja benar), lalu cari
        di zona mana tiap skor jatuh.
        """
        jitter = np.random.normal(
            0, config.THRESHOLD_JITTER_SD, size=len(self.base_thresholds)
        )
        thresholds_item = np.sort(self.base_thresholds + jitter)

        categories = np.searchsorted(thresholds_item, scores) + 1
        categories = np.clip(categories, config.LIKERT_MIN, config.LIKERT_MAX)
        return categories

    def convert_dataframe(self, df_continuous):
        """
        KENAPA:
        Ini yang dipanggil dari luar untuk mengonversi seluruh 40
        kolom sekaligus, satu per satu, tanpa harus menulis ulang
        loop di optimizer.py nanti.

        BAGAIMANA:
        Salin dulu strukturnya (df.copy()), lalu timpa tiap kolom
        dengan hasil konversi Likert-nya.
        """
        result = df_continuous.copy()
        for col in df_continuous.columns:
            result[col] = self.convert_one_column(df_continuous[col].values)
        return result


# ==========================================================
# BLOK PENGECEKAN MANDIRI
# ==========================================================
if __name__ == "__main__":
    from latent import LatentGenerator
    from structural import StructuralModel
    from indicator import IndicatorGenerator

    latent_gen = LatentGenerator()
    df_exogenous = latent_gen.sample_latent()
    df_latent = StructuralModel().apply(df_exogenous)

    indicator_gen = IndicatorGenerator(df_latent)
    df_continuous, loadings = indicator_gen.generate_all()

    ordinal = OrdinalEngine()
    df_likert = ordinal.convert_dataframe(df_continuous)

    print("=== CEK ORDINAL_ENGINE.PY ===")
    print("Tipe data setelah konversi (contoh kolom PE1):", df_likert["PE1"].dtype)

    print("\n5 baris pertama, kolom konstruk PE:")
    print(df_likert[config.CONSTRUCTS["PE"]].head())

    print(f"\nRata-rata keseluruhan (target {config.TARGET_MEAN}):",
          round(df_likert.values.mean(), 3))
    print(f"Simpangan baku keseluruhan (target {config.TARGET_STD}):",
          round(df_likert.values.std(), 3))

    print("\nDistribusi kategori untuk PE1 (harusnya condong ke 4-5):")
    print(df_likert["PE1"].value_counts().sort_index())
