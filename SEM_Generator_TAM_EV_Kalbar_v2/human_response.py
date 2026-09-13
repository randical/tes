"""
human_response.py
-----------------
Menyuntikkan ciri jawaban manusia yang tidak muncul dari model CFA murni.

Empat tombol, diurutkan dari yang paling murah ke yang paling mahal
biayanya terhadap indeks kecocokan.

1. Gaya akuiesen        biaya kecil sampai sedang, menaikkan varians
                        faktor tunggal pada uji Harman
2. Gaya ujung skala     biaya nyaris nol
3. Penjawab lurus       biaya BESAR, langsung menurunkan CFI
4. Data hilang          biaya tergantung prosedur estimasi

Prinsipnya, tiap tombol harus bisa dimatikan dan tiap biaya harus
diketahui sebelum dipakai. Realisme yang tidak terukur biayanya
bukan realisme, melainkan kerusakan yang tidak disengaja.
"""

import numpy as np

import config


class HumanResponseInjector:

    def __init__(self, seed=None):
        seed = seed if seed is not None else config.RANDOM_SEED + 4
        self.rng = np.random.default_rng(seed)
        self.log = {}

    # ----- tahap KONTINU, sebelum konversi Likert -----
    def apply_continuous(self, df_cont):
        """
        Gaya akuiesen adalah kecenderungan responden menjawab setuju
        terlepas dari isi pertanyaan. Efeknya satu pergeseran yang
        sama untuk SEMUA kolom pada responden tersebut. Karena berlaku
        lintas konstruk, inilah yang di literatur disebut faktor metode
        bersama, dan inilah yang seharusnya ditangkap uji Harman.

        Gaya ujung skala adalah kecenderungan sebagian responden
        memakai angka ekstrem sementara sebagian lain berkerumun di
        tengah. Efeknya pengali, bukan pergeseran.
        """
        hasil = df_cont.copy()
        n = len(hasil)

        if config.ACQUIESCENCE_SD > 0:
            pergeseran = self.rng.normal(0, config.ACQUIESCENCE_SD, size=n)
            hasil = hasil.add(pergeseran, axis=0)
            self.log["acquiescence_sd"] = config.ACQUIESCENCE_SD

        if config.EXTREME_STYLE_SD > 0:
            pengali = np.exp(self.rng.normal(0, config.EXTREME_STYLE_SD, size=n))
            hasil = hasil.mul(pengali, axis=0)
            self.log["extreme_style_sd"] = config.EXTREME_STYLE_SD

        return hasil

    # ----- tahap ORDINAL, setelah konversi Likert -----
    def apply_ordinal(self, df_likert):
        """
        Penjawab lurus adalah responden yang menekan angka yang sama
        untuk hampir semua pertanyaan. Data survei sungguhan selalu
        punya sedikit responden semacam ini. Mereka MERUSAK kecocokan
        model, jadi porsinya harus kecil dan disengaja.
        """
        hasil = df_likert.copy()
        n = len(hasil)
        items = list(hasil.columns)

        if config.STRAIGHTLINER_RATE > 0:
            jumlah = int(round(n * config.STRAIGHTLINER_RATE))
            if jumlah > 0:
                baris = self.rng.choice(n, size=jumlah, replace=False)
                for i in baris:
                    angka = int(self.rng.choice([config.SCALE_POINTS,
                                                 config.SCALE_POINTS - 1],
                                                p=[0.65, 0.35]))
                    nilai = np.full(len(items), angka)
                    # Sisakan satu dua penyimpangan supaya tidak identik persis
                    goyang = self.rng.choice(len(items),
                                             size=max(1, len(items) // 12),
                                             replace=False)
                    nilai[goyang] = np.clip(angka - 1, 1, config.SCALE_POINTS)
                    hasil.iloc[i, :] = nilai
                self.log["straightliners"] = jumlah

        if config.MISSING_RATE > 0:
            topeng = self.rng.random(hasil.shape) < config.MISSING_RATE
            hasil = hasil.mask(topeng)
            self.log["missing_cells"] = int(topeng.sum())

        return hasil


if __name__ == "__main__":
    from latent import LatentGenerator
    from structural import StructuralModel
    from indicator import IndicatorGenerator
    from ordinal_engine import OrdinalEngine

    df_full = StructuralModel().apply(LatentGenerator().sample())
    df_cont, loadings = IndicatorGenerator(df_full).generate_all()

    print("=== CEK HUMAN_RESPONSE.PY ===")
    print("Perbandingan korelasi rata-rata dalam konstruk PN\n")

    inj = HumanResponseInjector()
    df_gaya = inj.apply_continuous(df_cont)

    for label, d in [("tanpa gaya jawab", df_cont), ("dengan gaya jawab", df_gaya)]:
        dalam = d[config.CONSTRUCTS["PN"]].corr().to_numpy()
        dalam = dalam[np.triu_indices_from(dalam, k=1)].mean()
        antar = d[["PN1", "AP1", "CIA1", "PE1"]].corr().to_numpy()
        antar = antar[np.triu_indices_from(antar, k=1)].mean()
        print(f"  {label:<20} dalam konstruk {dalam:.3f}   antar konstruk {antar:.3f}")

    eng = OrdinalEngine()
    df_likert = eng.convert_dataframe(df_gaya)
    df_akhir = inj.apply_ordinal(df_likert)
    print("\n  catatan injeksi:", inj.log)
    print("  baris duplikat setelah injeksi:", int(df_akhir.duplicated().sum()))
    print("  rata-rata keseluruhan:", round(float(df_akhir.to_numpy().mean()), 3))
