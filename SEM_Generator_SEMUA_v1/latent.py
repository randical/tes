"""
latent.py
---------
Membangkitkan skor variabel laten (tersembunyi) untuk setiap
konstruk, lengkap dengan korelasi antar-konstruk yang realistis.

Modul ini BELUM menghasilkan data indikator (PE1, PE2, dst) --
itu tugas indicator.py. Modul ini baru menghasilkan skor konstruk
itu sendiri (PE, EE, SI, dst), dalam bentuk angka kontinu.
"""

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.stats import multivariate_normal

import config


class LatentGenerator:
    """
    Membungkus semua langkah untuk menghasilkan skor laten:
    membuat matriks korelasi, memperbaikinya kalau tidak valid,
    lalu membangkitkan skor acak yang mengikuti matriks tersebut.
    """

    def __init__(self, seed=None):
        # __init__ otomatis jalan begitu kita membuat objek baru,
        # misalnya: gen = LatentGenerator()
        # Di sinilah "bahan" yang dipakai berulang oleh method lain
        # di bawah disiapkan, supaya tidak perlu ditulis ulang tiap kali.
        #
        # seed bisa diisi dari luar (oleh optimizer.py, beda tiap
        # percobaan). Kalau tidak diisi (None), pakai bawaan config.py --
        # ini supaya file ini tetap bisa dites sendirian seperti sebelumnya.
        seed = seed if seed is not None else config.RANDOM_SEED
        np.random.seed(seed)
        # Konstruk endogen (CL) TIDAK ikut dibangkitkan di sini --
        # nilainya nanti dihitung oleh structural.py dari konstruk
        # eksogen lain. "[c for c in ... if c != ...]" ini disebut
        # list comprehension -- cara singkat menulis for-loop yang
        # menghasilkan list baru dengan syarat tertentu. Di sini
        # artinya: "ambil semua nama konstruk, KECUALI outcome-nya".
        all_constructs = list(config.CONSTRUCTS.keys())
        self.constructs = [c for c in all_constructs if c != config.STRUCTURAL_OUTCOME]
        self.n_construct = len(self.constructs)            # sekarang 7
        self.n_respondents = config.N_RESPONDENTS           # 200

    def create_correlation_matrix(self):
        """
        KENAPA:
        Manusia tidak punya hubungan antar-konstruk yang seragam --
        PE bisa berkorelasi 0.62 dengan EE, tapi cuma 0.41 dengan HM.
        Kalau semua korelasi dibuat sama, hasilnya terasa "buatan".
        Karena itu setiap pasangan konstruk diberi angka korelasi
        acak dalam rentang config.LATENT_CORR_MIN - LATENT_CORR_MAX.

        BAGAIMANA:
        Matriks berukuran 8x8. Diagonalnya (konstruk dengan dirinya
        sendiri) selalu 1. Sisanya diisi angka acak, lalu disalin ke
        posisi cerminnya (baris jadi kolom) supaya matriks simetris --
        korelasi PE-EE harus sama dengan korelasi EE-PE.
        """
        p = self.n_construct
        corr = np.eye(p)  # matriks identitas: diagonal 1, sisanya 0
        for i in range(p):
            for j in range(i + 1, p):
                value = np.random.uniform(
                    config.LATENT_CORR_MIN,
                    config.LATENT_CORR_MAX
                )
                corr[i, j] = value
                corr[j, i] = value  # jaga simetri
        return corr

    def nearest_psd(self, matrix):
        """
        KENAPA:
        Tidak semua tabel angka yang "kelihatannya" seperti matriks
        korelasi valid dipakai untuk membangkitkan data. Kalau tidak
        valid, scipy akan menolak dengan error semacam
        "covariance matrix is not positive semi-definite".
        Ini sering terjadi kalau korelasi antar-pasangan diisi acak
        tanpa pengecekan ulang. Function ini memperbaikinya secara
        otomatis sebelum dipakai.

        CATATAN PENCEGAHAN:  # <-- BARU
        Setelah eigenvalue negatif "dinaikkan" jadi sedikit positif,
        hasilnya BELUM TENTU diagonalnya tetap tepat 1.0 -- padahal
        syarat matriks KORELASI (bukan sekadar kovarians) adalah
        diagonal harus selalu 1. Kalau tidak dijamin, korelasi antar
        konstruk yang keluar bisa melenceng dari rentang
        LATENT_CORR_MIN/MAX yang dirancang di config.py, tanpa
        ketahuan. Di data Anda kemarin ini TIDAK terpicu (matriksnya
        kebetulan sudah valid), tapi tetap dijaga untuk kasus lain.

        BAGAIMANA:
        Tidak perlu dihafal rumusnya -- cukup paham bahwa 3 baris di
        bawah "membongkar" matriks jadi komponen-komponennya, menaikkan
        bagian yang bernilai negatif (secara statistik mustahil) jadi
        sedikit positif, lalu menyusunnya kembali jadi matriks yang sah.
        Baris tambahan di bawah (d = ..., matrix_fixed / ...) memaksa
        diagonal kembali tepat 1.0 -- proses ini disebut mengubah
        matriks KOVARIANS menjadi matriks KORELASI.
        """
        eigval, eigvec = eigh(matrix)
        eigval[eigval < 1e-8] = 1e-8
        matrix_fixed = eigvec @ np.diag(eigval) @ eigvec.T

        # Renormalisasi -- jamin diagonal tepat 1.0 (matriks korelasi sah)  # <-- BARU
        d = np.sqrt(np.diag(matrix_fixed))
        matrix_fixed = matrix_fixed / np.outer(d, d)
        np.fill_diagonal(matrix_fixed, 1.0)  # jaga-jaga sisa pembulatan floating point

        return matrix_fixed

    def sample_latent(self):
        """
        KENAPA:
        Ini bagian inti: membangkitkan skor laten untuk 200 responden
        x 8 konstruk sekaligus, mengikuti pola korelasi yang sudah
        dirancang di atas -- bukan angka acak yang saling lepas.

        BAGAIMANA:
        scipy punya fungsi siap pakai, multivariate_normal, yang bisa
        membangkitkan banyak variabel acak sekaligus mengikuti sebuah
        matriks korelasi tertentu. Hasilnya dibungkus jadi tabel
        (DataFrame) dengan nama kolom PE, EE, SI, dst supaya mudah dibaca.
        """
        corr = self.create_correlation_matrix()
        corr = self.nearest_psd(corr)

        mean = np.zeros(self.n_construct)  # rata-rata tiap konstruk = 0
        latent_scores = multivariate_normal.rvs(
            mean=mean,
            cov=corr,
            size=self.n_respondents
        )

        df = pd.DataFrame(latent_scores, columns=self.constructs)
        return df


# ==========================================================
# BLOK PENGECEKAN MANDIRI
# ==========================================================
# Sama seperti config.py: blok ini hanya jalan kalau file ini
# dibuka langsung (python latent.py), bukan saat di-import.
if __name__ == "__main__":
    generator = LatentGenerator()
    df_latent = generator.sample_latent()

    print("=== CEK LATENT.PY ===")
    print(f"Ukuran data: {df_latent.shape[0]} responden x {df_latent.shape[1]} konstruk")
    print("\nRata-rata tiap konstruk (idealnya mendekati 0):")
    print(df_latent.mean().round(3))
    print(f"\nKorelasi antar konstruk (target: {config.LATENT_CORR_MIN}-{config.LATENT_CORR_MAX}):")
    print(df_latent.corr().round(2))
