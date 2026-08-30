"""
demographics.py
---------------
Membangkitkan kolom demografi.

KENAPA TIDAK CUKUP MENGUNDI SETIAP KOLOM SECARA BEBAS
Kalau tiap kolom diundi independen, tabulasi silang pendidikan lawan
pendapatan akan mendekati sempurna acak. Reviewer yang memeriksa tabel
demografi akan melihat pola yang terlalu rapi dan tidak masuk akal.

CARA YANG DIPAKAI
Satu faktor laten status sosial ekonomi dibangkitkan lebih dulu.
Tiap variabel demografi diberi bobot terhadap faktor itu lewat
ses_loading di config. Bobot nol berarti benar-benar bebas, misalnya
jenis kelamin. Bobot tinggi berarti sangat terkait, misalnya pendapatan.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

import config


class DemographicGenerator:

    def __init__(self, seed=None):
        seed = seed if seed is not None else config.RANDOM_SEED + 5
        self.rng = np.random.default_rng(seed)
        self.codes = None

    def generate(self, n=None, df_latent=None):
        """
        df_latent opsional. Kalau diberikan, sebuah variabel demografi bisa
        dijangkarkan ke konstruk laten lewat kunci "construct_loading".

        KENAPA PENJANGKARAN INI PENTING:
        Variabel kontrol yang benar-benar tidak berkorelasi dengan apa pun
        justru tidak realistis. Di data nyata, usia dan pendapatan hampir
        selalu punya kaitan lemah dengan sikap yang diteliti. Tanpa kaitan
        itu, koefisien kontrol akan selalu mendarat di sekitar nol dan
        latihan menafsirkan variabel kontrol jadi tidak ada isinya.
        """
        n = n or config.N_RESPONDENTS
        ses = self.rng.normal(0, 1, n)          # faktor status sosial ekonomi
        kolom = {}     # berisi LABEL teks, untuk tabel demografi
        kode = {}      # berisi KODE angka 1..k, untuk kovariat dan pengelompokan

        for nama, spec in config.DEMOGRAPHIC_SPEC.items():
            w = float(spec.get("ses_loading", 0.0))
            jangkar = np.zeros(n)
            w_konstruk = 0.0
            for konstruk, bobot in spec.get("construct_loading", {}).items():
                if df_latent is None or konstruk not in df_latent.columns:
                    raise ValueError(
                        f"'{nama}' minta dijangkarkan ke '{konstruk}', tapi skor "
                        "laten konstruk itu belum tersedia. Penjangkaran hanya "
                        "bisa ke konstruk EKSOGEN, karena demografi dibangkitkan "
                        "sebelum tahap struktural dijalankan.")
                skor = df_latent[konstruk].to_numpy(dtype=float)
                jangkar = jangkar + float(bobot) * (skor - skor.mean()) / skor.std()
                w_konstruk += float(bobot) ** 2

            sisa = 1.0 - w ** 2 - w_konstruk
            if sisa <= 0:
                raise ValueError(
                    f"Bobot untuk '{nama}' terlalu besar. Kuadrat ses_loading "
                    "ditambah kuadrat seluruh construct_loading harus di bawah 1.")
            laten = w * ses + jangkar + np.sqrt(sisa) * self.rng.normal(0, 1, n)

            probs = np.array(spec["probs"], dtype=float)
            probs = probs / probs.sum()
            batas = norm.ppf(np.cumsum(probs)[:-1])
            idx = np.searchsorted(batas, laten)
            kolom[nama] = np.array(spec["labels"])[idx]
            kode[nama] = idx + 1          # 1..k, urut sesuai daftar labels

        self.codes = pd.DataFrame(kode)
        return pd.DataFrame(kolom)


if __name__ == "__main__":
    df = DemographicGenerator().generate()
    print("=== CEK DEMOGRAPHICS.PY ===")
    print("Ukuran:", df.shape, "\n")
    for kol in df.columns:
        vc = df[kol].value_counts(normalize=True).sort_index()
        target = dict(zip(config.DEMOGRAPHIC_SPEC[kol]["labels"],
                          config.DEMOGRAPHIC_SPEC[kol]["probs"]))
        print(f"{kol}")
        for label, prop in vc.items():
            print(f"    {label:<22}{prop:>7.3f}   target {target.get(label, 0):.3f}")

    print("\nTabulasi silang Pendidikan lawan Pendapatan (proporsi baris)")
    silang = pd.crosstab(df["Education"], df["Income"], normalize="index")
    print(silang.round(3).to_string())
    print("\nKalau kedua kolom diundi bebas, tiap baris akan hampir sama.")
    print("Kenaikan proporsi kolom >10jt dari baris SMA ke Pascasarjana")
    print("adalah bukti keterkaitan status sosial ekonomi bekerja.")
