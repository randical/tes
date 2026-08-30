"""
indicator.py
------------
Mengubah skor konstruk laten (PE, EE, SI, dst) menjadi 40 skor
indikator (PE1, PE2, ..., CL5) yang masih berupa angka kontinu.

Rumus dasarnya, sesuai teori SEM/CFA:
    indikator = loading x skor_laten + error

Modul ini BELUM mengubah angka jadi skala Likert 1-5 -- itu tugas
ordinal_engine.py.
"""

import numpy as np
import pandas as pd

import config


class IndicatorGenerator:
    """
    Membungkus langkah-langkah untuk mengubah skor laten menjadi
    skor indikator: menentukan loading tiap item, menghitung error
    yang sesuai, lalu menggabungkan keduanya jadi satu skor.
    """

    def __init__(self, latent_df, seed=None):
        # latent_df adalah hasil dari latent.py (tabel 200 x 8).
        # Disimpan di self supaya semua method di bawah bisa
        # memakainya tanpa perlu dioper ulang setiap kali dipanggil.
        seed = seed if seed is not None else config.RANDOM_SEED + 1
        np.random.seed(seed)
        self.latent_df = latent_df
        self.n_respondents = len(latent_df)
        self.constructs = config.CONSTRUCTS  # dictionary lengkap dari config.py

    def generate_loadings(self):
        """
        KENAPA:
        Loading factor menunjukkan seberapa kuat satu item "mewakili"
        konstruknya. Kalau semua item dalam satu konstruk diberi
        loading yang sama persis, data terasa kaku -- di survei
        sungguhan, PE1 biasanya sedikit lebih kuat mewakili PE
        dibanding PE3. Karena itu tiap item diberi loading acak
        dalam rentang config.LOADING_MIN - LOADING_MAX.

        BAGAIMANA:
        Untuk tiap konstruk, untuk tiap item di dalamnya, simpan satu
        angka loading acak. Hasilnya disimpan sebagai "dictionary di
        dalam dictionary" -- loading["PE"]["PE1"] berisi angka loading
        khusus untuk item PE1.
        """
        loading = {}
        for construct, items in self.constructs.items():
            loading[construct] = {}
            for item in items:
                loading[construct][item] = np.random.uniform(
                    config.LOADING_MIN,
                    config.LOADING_MAX
                )
        return loading

    def calculate_error_sigma(self, loading_value):
        """
        KENAPA:
        Dalam teori CFA, kekuatan item terhadap konstruknya dan
        "derau" (noise) yang menyertainya saling melengkapi -- totalnya
        harus 1. Jadi kalau loading tinggi (item kuat mewakili
        konstruk), error harus kecil, dan sebaliknya. ini bukan angka
        yang kita tebak sembarangan, tapi dihitung dari loading-nya.

        BAGAIMANA:
        Variansi error = 1 - loading^2. Diambil akarnya (sqrt) supaya
        dapat "sigma" (simpangan baku) yang bisa langsung dipakai oleh
        np.random.normal. Nilai minimum 0.05 dipasang sebagai
        jaga-jaga supaya error tidak pernah nol sama sekali.
        """
        variance = max(1 - loading_value ** 2, 0.05)
        return np.sqrt(variance)

    def generate_one_indicator(self, latent_score, loading_value):
        """
        KENAPA:
        Ini rumus inti SEM: skor indikator = loading x skor_laten + error.
        Errornya harus UNIK untuk setiap responden -- bukan satu angka
        error yang dipakai untuk semua 200 responden -- supaya tidak
        ada dua responden yang menjawab persis sama. Sigma-nya juga
        digoyang sedikit per responden (heteroskedastisitas) karena
        manusia tidak semuanya sekonsisten saat mengisi survei.

        BAGAIMANA:
        sigma dasar dihitung dari loading, lalu digoyang 90%-110%
        khusus untuk tiap responden, baru dipakai membangkitkan 200
        angka error yang saling berbeda sekaligus (bukan satu-satu).
        """
        sigma_dasar = self.calculate_error_sigma(loading_value)
        sigma_per_responden = sigma_dasar * np.random.uniform(
            0.90, 1.10, self.n_respondents
        )
        error = np.random.normal(0, sigma_per_responden, self.n_respondents)
        return loading_value * latent_score + error

    def generate_all(self):
        """
        KENAPA:
        Ini yang menjalankan semuanya untuk 40 indikator sekaligus --
        nanti dipanggil satu kali oleh optimizer.py untuk menghasilkan
        tabel data indikator yang lengkap.

        BAGAIMANA:
        Untuk tiap konstruk dan tiap item di dalamnya: ambil skor
        laten konstruknya, ambil loading item tersebut, panggil
        generate_one_indicator(). Hasilnya dikumpulkan jadi satu
        dictionary, lalu diubah jadi tabel (DataFrame).
        """
        loading = self.generate_loadings()
        data = {}

        for construct, items in self.constructs.items():
            latent_score = self.latent_df[construct]
            for item in items:
                data[item] = self.generate_one_indicator(
                    latent_score, loading[construct][item]
                )

        df_indicators = pd.DataFrame(data)
        return df_indicators, loading


# ==========================================================
# BLOK PENGECEKAN MANDIRI
# ==========================================================
if __name__ == "__main__":
    from latent import LatentGenerator
    from structural import StructuralModel

    latent_gen = LatentGenerator()
    df_exogenous = latent_gen.sample_latent()
    df_latent = StructuralModel().apply(df_exogenous)

    indicator_gen = IndicatorGenerator(df_latent)
    df_indicators, loadings = indicator_gen.generate_all()

    print("=== CEK INDICATOR.PY ===")
    print(f"Ukuran data: {df_indicators.shape[0]} responden x {df_indicators.shape[1]} indikator")

    print("\nContoh loading factor konstruk PE:")
    for item, value in loadings["PE"].items():
        print(f"  {item}: {value:.3f}")

    pe_items = config.CONSTRUCTS["PE"]
    print("\nKorelasi antar item dalam konstruk PE (harusnya positif, bukan 0):")
    print(df_indicators[pe_items].corr().round(2))
