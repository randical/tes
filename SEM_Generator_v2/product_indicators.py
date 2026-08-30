"""
product_indicators.py
---------------------
Membangun kolom indikator produk untuk uji INTERAKSI LATEN di AMOS.

KENAPA MODUL INI ADA
AMOS tidak punya fasilitas otomatis untuk membuat konstruk interaksi
laten seperti perintah XWITH di Mplus. Cara baku di AMOS adalah
membuat konstruk laten baru yang indikatornya berupa PERKALIAN antar
indikator kedua konstruk induk, lalu menggambarnya sebagai prediktor
tambahan. Kolom perkaliannya harus sudah tersedia di data.
[Keyakinan tinggi untuk ketiadaan fasilitas otomatis. Kalau versi
AMOS Anda ternyata sudah menyediakannya, modul ini tinggal dimatikan.]

STRATEGI PASANGAN TERCOCOK
Kalau semua item konstruk A dikalikan semua item konstruk B, lima kali
lima menghasilkan dua puluh lima indikator produk. Jumlah itu membuat
model membengkak dan error-nya saling tumpang tindih berat. Strategi
pasangan tercocok mengurutkan item tiap konstruk menurut kekuatan
loading-nya, lalu memasangkan satu lawan satu. Hasilnya sebanyak item
konstruk terkecil saja. Tiap item induk terpakai tepat sekali,
sehingga tumpang tindih error jauh berkurang.

PEMUSATAN RERATA GANDA
Tiap item dipusatkan lebih dulu, dikalikan, lalu hasil perkaliannya
dipusatkan sekali lagi. Langkah kedua menekan korelasi
antara konstruk interaksi dan efek utamanya. Perlu dicatat jujur,
penekanan itu TIDAK sempurna. Ortogonalitas penuh hanya berlaku kalau
kedua variabel berdistribusi normal bersama, sedangkan data Likert yang
condong tidak memenuhi syarat itu. Pada pengujian sandbox, sisa
korelasinya sekitar 0,12 sampai 0,14. Angka sebesar itu masih jauh di
bawah taraf kolinearitas yang mengganggu estimasi, tapi jangan
dilaporkan sebagai nol.
"""

import numpy as np
import pandas as pd

import config


def matched_pair_products(df_likert, predictor, moderator,
                          loadings_hint=None, prefix=None):
    items_p = list(config.CONSTRUCTS[predictor])
    items_m = list(config.CONSTRUCTS[moderator])

    def urutkan(items):
        if loadings_hint:
            pasangan = [(it, loadings_hint.get(it, 0.0)) for it in items]
        else:
            # Tanpa petunjuk loading, pakai korelasi item terhadap
            # skor komposit konstruknya sebagai pengganti kasar.
            komposit = df_likert[items].astype(float).mean(axis=1)
            pasangan = [(it, float(df_likert[it].astype(float).corr(komposit)))
                        for it in items]
        return [it for it, _ in sorted(pasangan, key=lambda x: -x[1])]

    urut_p, urut_m = urutkan(items_p), urutkan(items_m)
    k = min(len(urut_p), len(urut_m))
    prefix = prefix or f"{predictor}x{moderator}"

    hasil = {}
    for i in range(k):
        a = df_likert[urut_p[i]].astype(float)
        b = df_likert[urut_m[i]].astype(float)
        produk = (a - a.mean()) * (b - b.mean())     # pemusatan pertama
        produk = produk - produk.mean()              # pemusatan kedua
        hasil[f"{prefix}_{i + 1}"] = produk.round(4)
    return pd.DataFrame(hasil, index=df_likert.index), list(zip(urut_p[:k], urut_m[:k]))


def build_all(df_likert, design_loadings=None):
    if not getattr(config, "BUILD_PRODUCT_INDICATORS", False):
        return pd.DataFrame(index=df_likert.index), {}

    petunjuk = {}
    if design_loadings:
        for d in design_loadings.values():
            petunjuk.update(d)

    semua, catatan = [], {}
    for m in config.MODERATIONS:
        blok, pasangan = matched_pair_products(
            df_likert, m["predictor"], m["moderator"], petunjuk or None)
        semua.append(blok)
        catatan[f'{m["predictor"]}x{m["moderator"]}'] = pasangan
    gabung = pd.concat(semua, axis=1) if semua else pd.DataFrame(index=df_likert.index)
    return gabung, catatan


if __name__ == "__main__":
    from latent import LatentGenerator
    from structural import StructuralModel
    from indicator import IndicatorGenerator
    from ordinal_engine import OrdinalEngine
    from human_response import HumanResponseInjector

    full = StructuralModel().apply(LatentGenerator().sample())
    gen = IndicatorGenerator(full)
    cont, loadings = gen.generate_all()
    inj = HumanResponseInjector()
    likert = inj.apply_ordinal(OrdinalEngine().convert_dataframe(
        inj.apply_continuous(cont)))

    blok, catatan = build_all(likert, loadings)

    print("=== CEK PRODUCT_INDICATORS.PY ===")
    print("Kolom produk:", list(blok.columns))
    for nama, pasangan in catatan.items():
        print(f"\nPasangan tercocok untuk {nama}")
        for i, (a, b) in enumerate(pasangan, 1):
            print(f"    {nama}_{i} = terpusat({a}) x terpusat({b})")

    komposit_pn = likert[config.CONSTRUCTS["PN"]].astype(float).mean(axis=1)
    komposit_cia = likert[config.CONSTRUCTS["CIA"]].astype(float).mean(axis=1)
    komposit_prod = blok.mean(axis=1)

    print("\nUji ortogonalitas hasil pemusatan ganda")
    print(f"    korelasi produk dengan PN  : {komposit_prod.corr(komposit_pn):+.4f}")
    print(f"    korelasi produk dengan CIA : {komposit_prod.corr(komposit_cia):+.4f}")
    print("    (sisa korelasi kecil memang wajar pada data Likert yang condong,")
    print("     yang penting jauh di bawah taraf kolinearitas mengganggu)")
