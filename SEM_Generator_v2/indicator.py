"""
indicator.py
------------
Mengubah skor konstruk laten menjadi skor indikator kontinu, mengikuti
persamaan dasar CFA.

    indikator = loading x skor_laten + error

Belum ada skala Likert di sini. Itu tugas ordinal_engine.py.
"""

import numpy as np
import pandas as pd

import config


class IndicatorGenerator:

    def __init__(self, df_latent, seed=None):
        seed = seed if seed is not None else config.RANDOM_SEED + 2
        self.rng = np.random.default_rng(seed)
        self.df_latent = df_latent
        self.n = len(df_latent)

    def generate_loadings(self):
        """
        KENAPA:
        Kalau semua item dalam satu konstruk diberi loading identik,
        data terasa kaku. Di survei sungguhan selalu ada item yang
        sedikit lebih kuat mewakili konstruknya dibanding item lain.

        BAGAIMANA:
        Undi satu loading acak per item dalam rentang LOADING_MIN
        sampai LOADING_MAX. Struktur hasilnya kamus di dalam kamus,
        persis seperti objek JSON bersarang.
        """
        loading = {}
        for konstruk, items in config.CONSTRUCTS.items():
            loading[konstruk] = {
                item: float(self.rng.uniform(config.LOADING_MIN, config.LOADING_MAX))
                for item in items
            }
        return loading

    @staticmethod
    def error_sigma(loading_value):
        """
        Dalam CFA terstandardisasi, variansi item dipecah jadi bagian
        yang dijelaskan konstruk (loading kuadrat) dan bagian error.
        Totalnya satu. Jadi sigma error bukan tebakan, tapi turunan
        langsung dari loading-nya.
        """
        return float(np.sqrt(max(1.0 - loading_value ** 2, 0.0025)))

    def generate_crossloadings(self):
        """
        KENAPA:
        Struktur faktor yang bersih sempurna adalah tanda data rekayasa.
        Tanpa muatan silang, CFI data buatan menyentuh 0.999, angka yang
        praktis tidak pernah muncul di survei sungguhan berukuran puluhan
        indikator. Item nyata selalu sedikit "bocor" ke konstruk tetangga
        karena redaksi tidak pernah sepenuhnya murni.

        BAGAIMANA:
        Sebagian item dipilih acak, lalu diberi satu muatan kecil ke satu
        konstruk lain yang dipilih acak. Besarnya diundi dari sebaran
        normal ber-simpangan CROSSLOADING_SD, bisa positif atau negatif.
        """
        silang = {}
        if getattr(config, "CROSSLOADING_RATE", 0) <= 0:
            return silang
        semua = list(config.CONSTRUCTS)
        for konstruk, items in config.CONSTRUCTS.items():
            lain = [c for c in semua if c != konstruk]
            for item in items:
                if self.rng.random() < config.CROSSLOADING_RATE:
                    target = str(self.rng.choice(lain))
                    silang[item] = (target,
                                    float(self.rng.normal(0, config.CROSSLOADING_SD)))
        return silang

    def _correlated_error_pairs(self):
        """
        Pasangan item sekonstruk yang berbagi redaksi mirip biasanya
        punya error berkorelasi. Di AMOS inilah panah dua arah antar
        error term yang sering ditambahkan lewat Modification Indices.
        Kalau data buatan tidak punyanya, latihan membaca Modification
        Indices jadi tidak ada gunanya.
        """
        pasangan = list(getattr(config, "FORCED_ERROR_PAIRS", []))
        if getattr(config, "CORRELATED_ERROR_RATE", 0) <= 0:
            return pasangan
        for items in config.CONSTRUCTS.values():
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    if (items[i], items[j]) in pasangan:
                        continue
                    if self.rng.random() < config.CORRELATED_ERROR_RATE:
                        pasangan.append((items[i], items[j]))
        return pasangan

    def generate_all(self):
        loadings = self.generate_loadings()
        silang = self.generate_crossloadings()
        pasangan_error = self._correlated_error_pairs()

        errors, kolom_lam = {}, {}
        for konstruk, items in config.CONSTRUCTS.items():
            for item in items:
                lam = loadings[konstruk][item]
                kolom_lam[item] = lam
                errors[item] = self.rng.normal(0, 1.0, size=self.n)

        # Suntikkan korelasi antar error pasangan terpilih
        r = getattr(config, "CORRELATED_ERROR_R", 0.0)
        for a, b in pasangan_error:
            errors[b] = r * errors[a] + np.sqrt(1 - r ** 2) * errors[b]

        # Heteroskedastisitas per responden. Satu pengali sigma untuk
        # tiap responden, dipakai konsisten di seluruh item, karena yang
        # bervariasi adalah konsistensi ORANGNYA, bukan konsistensi item.
        j = getattr(config, "ERROR_SIGMA_JITTER", 0.0)
        if j > 0:
            pengali_sigma = self.rng.uniform(1 - j, 1 + j, size=self.n)
        else:
            pengali_sigma = np.ones(self.n)
        self.sigma_multiplier = pengali_sigma

        kolom = {}
        for konstruk, items in config.CONSTRUCTS.items():
            skor_laten = self.df_latent[konstruk].to_numpy(dtype=float)
            for item in items:
                lam = kolom_lam[item]
                nilai = lam * skor_laten
                if item in silang:
                    target, beta = silang[item]
                    nilai = nilai + beta * self.df_latent[target].to_numpy(dtype=float)
                kolom[item] = nilai + self.error_sigma(lam) * pengali_sigma * errors[item]

        self.crossloadings = silang
        self.correlated_error_pairs = pasangan_error
        return pd.DataFrame(kolom, index=self.df_latent.index), loadings


if __name__ == "__main__":
    from latent import LatentGenerator
    from structural import StructuralModel

    df_full = StructuralModel().apply(LatentGenerator().sample())
    gen_last = IndicatorGenerator(df_full)
    df_cont, loadings = gen_last.generate_all()

    print("=== CEK INDICATOR.PY ===")
    print("Ukuran:", df_cont.shape)
    for k, d in loadings.items():
        nilai = np.array(list(d.values()))
        print(f"  {k:<4} loading rancangan {nilai.min():.3f} - {nilai.max():.3f} "
              f"(rata-rata {nilai.mean():.3f})")

    print(f"\nMuatan silang: {len(gen_last.crossloadings)} item")
    for it, (tg, b) in list(gen_last.crossloadings.items())[:5]:
        print(f"    {it} -> {tg}  beta {b:+.3f}")
    print(f"Pasangan error berkorelasi: {len(gen_last.correlated_error_pairs)}")
    sm = gen_last.sigma_multiplier
    print(f"Pengali sigma error per responden: {sm.min():.3f} sampai {sm.max():.3f}")

    print("\nKorelasi item dalam konstruk PN (harus tinggi):")
    print(df_cont[config.CONSTRUCTS["PN"]].corr().round(3).to_string())
    print("\nKorelasi PN1 dengan item konstruk lain (harus jauh lebih rendah):")
    print(df_cont[["PN1", "AP1", "CIA1", "PE1"]].corr().round(3).loc["PN1"].to_string())
