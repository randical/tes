"""
latent.py
---------
Membangkitkan skor konstruk EKSOGEN, yaitu konstruk yang tidak punya
panah masuk di STRUCTURAL_MODEL.

Konstruk endogen TIDAK dibangkitkan di sini. Itu tugas structural.py,
karena nilainya harus dihitung dari prediktornya, bukan diundi.
"""

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.stats import multivariate_normal

import config


class LatentGenerator:

    def __init__(self, seed=None, constructs=None):
        seed = seed if seed is not None else config.RANDOM_SEED
        self.rng = np.random.default_rng(seed)
        self.constructs = constructs or config.exogenous_constructs()
        self.n = len(self.constructs)

    def create_correlation_matrix(self):
        """
        KENAPA:
        Konstruk sikap di dunia nyata tidak pernah saling lepas. Orang
        yang skor Social Influence-nya tinggi cenderung juga punya
        Performance Expectancy yang tidak nol. Kalau semua konstruk
        dibangkitkan independen, matriks korelasinya akan mendekati
        matriks identitas dan data terasa mekanis.

        BAGAIMANA:
        Undi satu angka acak untuk tiap PASANGAN konstruk dalam rentang
        LATENT_CORR_MIN sampai LATENT_CORR_MAX, lalu cerminkan ke sisi
        bawah matriks. Diagonal selalu 1.0.
        """
        override = getattr(config, "LATENT_CORR_OVERRIDE", None)
        m = np.eye(self.n)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                a, b = self.constructs[i], self.constructs[j]
                if override is not None and ((a, b) in override or (b, a) in override):
                    r = float(override.get((a, b), override.get((b, a))))
                else:
                    r = self.rng.uniform(config.LATENT_CORR_MIN, config.LATENT_CORR_MAX)
                m[i, j] = m[j, i] = r
        return m

    @staticmethod
    def nearest_psd(matrix):
        """
        KENAPA:
        Mengundi tiap pasangan secara independen bisa menghasilkan
        kombinasi yang mustahil secara geometris. Contoh klasik, kalau
        A-B = 0.9 dan A-C = 0.9, maka B-C tidak mungkin 0.1. Matriks
        semacam itu punya eigenvalue negatif dan scipy akan menolak
        memakainya untuk membangkitkan data.

        BAGAIMANA:
        Bongkar matriks jadi eigenvalue dan eigenvector, naikkan
        eigenvalue negatif menjadi angka positif sangat kecil, lalu
        susun ulang.

        LANGKAH TERAKHIR YANG SERING DILUPAKAN:
        Setelah disusun ulang, diagonalnya tidak lagi dijamin tepat 1.0.
        Hasilnya jadi matriks KOVARIANS, bukan KORELASI. Dua baris
        renormalisasi di bawah mengembalikannya. Tanpa ini, seluruh
        koefisien jalur hilir akan bergeser diam-diam.
        """
        eigval, eigvec = eigh(matrix)
        eigval[eigval < 1e-8] = 1e-8
        fixed = eigvec @ np.diag(eigval) @ eigvec.T

        d = np.sqrt(np.diag(fixed))
        fixed = fixed / np.outer(d, d)
        np.fill_diagonal(fixed, 1.0)
        return fixed

    def sample(self):
        corr = self.nearest_psd(self.create_correlation_matrix())
        skor = multivariate_normal.rvs(
            mean=np.zeros(self.n),
            cov=corr,
            size=config.N_RESPONDENTS,
            random_state=int(self.rng.integers(0, 2**31 - 1)),
        )
        skor = np.atleast_2d(skor)
        if skor.shape[0] != config.N_RESPONDENTS:
            skor = skor.T
        return pd.DataFrame(skor, columns=self.constructs)


if __name__ == "__main__":
    gen = LatentGenerator()
    df = gen.sample()
    print("=== CEK LATENT.PY ===")
    print("Konstruk eksogen :", gen.constructs)
    print("Ukuran           :", df.shape)
    print("\nRata-rata (idealnya mendekati 0):")
    print(df.mean().round(3).to_string())
    print("\nSimpangan baku (idealnya mendekati 1):")
    print(df.std().round(3).to_string())
    print("\nKorelasi antar konstruk:")
    print(df.corr().round(3).to_string())
