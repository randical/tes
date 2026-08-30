"""
validate_cfa.py
---------------
LAPIS DUA. Pemeriksa presisi. Dijalankan SEKALI setelah main.py,
sebelum data dibuka di AMOS.

KENAPA LAPIS INI TIDAK BISA DIHAPUS
quality_control.py memakai estimasi komponen utama karena harus cepat.
Estimasi itu BIAS KE ATAS. Komponen utama menyerap sebagian variansi
unik item, sehingga loading yang dilaporkannya lebih tinggi dari
loading CFA sungguhan. Artinya lapis satu bersifat OPTIMIS, dan
optimisme adalah arah bias yang paling berbahaya untuk sebuah
saringan kelayakan.

Modul ini menjalankan CFA sungguhan lewat semopy, lalu menghitung
ulang AVE, CR, dan validitas diskriminan dari loading dan korelasi
laten hasil estimasi, bukan dari pendekatan apa pun.
"""

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import config


def build_cfa_spec(constructs=None):
    """Menyusun sintaks model semopy dari config.CONSTRUCTS."""
    constructs = constructs or config.CONSTRUCTS
    baris = [f"{k} =~ " + " + ".join(items) for k, items in constructs.items()]
    return "\n".join(baris)


def run_cfa(df, constructs=None):
    from semopy import Model

    spec = build_cfa_spec(constructs)
    model = Model(spec)
    model.fit(df)
    return model


def standardized_loadings(model, constructs=None):
    constructs = constructs or config.CONSTRUCTS
    ins = model.inspect(std_est=True)
    kolom_std = "Est. Std" if "Est. Std" in ins.columns else "Estimate"
    muat = {}
    for konstruk, items in constructs.items():
        nilai = []
        for item in items:
            baris = ins[(ins["lval"] == item) & (ins["op"] == "~")
                        & (ins["rval"] == konstruk)]
            if len(baris) == 0:
                baris = ins[(ins["lval"] == konstruk) & (ins["op"] == "=~")
                            & (ins["rval"] == item)]
            nilai.append(abs(float(baris[kolom_std].iloc[0])))
        muat[konstruk] = np.array(nilai)
    return muat


def latent_correlations(model, constructs=None):
    constructs = constructs or config.CONSTRUCTS
    nama = list(constructs)
    ins = model.inspect(std_est=True)
    kolom_std = "Est. Std" if "Est. Std" in ins.columns else "Estimate"
    mat = pd.DataFrame(np.eye(len(nama)), index=nama, columns=nama)
    for i, a in enumerate(nama):
        for b in nama[i + 1:]:
            baris = ins[(ins["op"] == "~~")
                        & (((ins["lval"] == a) & (ins["rval"] == b))
                           | ((ins["lval"] == b) & (ins["rval"] == a)))]
            if len(baris):
                mat.loc[a, b] = mat.loc[b, a] = float(baris[kolom_std].iloc[0])
    return mat


def cr_ave(loading):
    sum_l = loading.sum()
    sum_err = (1 - loading ** 2).sum()
    return float((sum_l ** 2) / ((sum_l ** 2) + sum_err)), float((loading ** 2).mean())


def fit_indices(model):
    from semopy import calc_stats
    stats = calc_stats(model).T
    seri = stats.iloc[:, 0]
    ambil = ["chi2", "DoF", "chi2 p-value", "CFI", "TLI", "RMSEA", "GFI", "AGFI", "NFI"]
    return {k: (round(float(seri[k]), 4) if k in seri.index else None) for k in ambil}


def report(df, constructs=None, verbose=True):
    constructs = constructs or config.CONSTRUCTS
    items = [i for v in constructs.values() for i in v]
    model = run_cfa(df[items].astype(float), constructs)

    muat = standardized_loadings(model, constructs)
    korelasi = latent_correlations(model, constructs)
    fit = fit_indices(model)

    tabel = {}
    for k, l in muat.items():
        cr, ave = cr_ave(l)
        tabel[k] = {"loading_min": round(float(l.min()), 3),
                    "loading_mean": round(float(l.mean()), 3),
                    "CR": round(cr, 3), "AVE": round(ave, 3),
                    "akar_AVE": round(float(np.sqrt(ave)), 3)}

    diskriminan = {}
    for a in constructs:
        r = korelasi.loc[a].drop(a).abs()
        diskriminan[a] = {"akar_AVE": tabel[a]["akar_AVE"],
                          "korelasi_tertinggi": round(float(r.max()), 3),
                          "lawan": str(r.idxmax()),
                          "passed": tabel[a]["akar_AVE"] > float(r.max())}

    if verbose:
        print("=== LAPIS DUA, CFA SUNGGUHAN (semopy) ===\n")
        print("Indeks kecocokan")
        for k, v in fit.items():
            print(f"  {k:<14}{v}")
        if fit["DoF"]:
            print(f"  {'CMIN/DF':<14}{round(fit['chi2'] / fit['DoF'], 4)}")

        print("\nModel pengukuran")
        print(f"{'Konstruk':<8}{'load min':>10}{'load rata':>11}{'CR':>8}{'AVE':>8}")
        for k, v in tabel.items():
            print(f"{k:<8}{v['loading_min']:>10.3f}{v['loading_mean']:>11.3f}"
                  f"{v['CR']:>8.3f}{v['AVE']:>8.3f}")

        print("\nValiditas diskriminan Fornell-Larcker")
        for k, v in diskriminan.items():
            print(f"  {k:<5} akar AVE {v['akar_AVE']:.3f} vs korelasi laten "
                  f"{v['korelasi_tertinggi']:.3f} ({v['lawan']}) "
                  f"-> {'OK' if v['passed'] else 'GAGAL'}")

        print("\nKorelasi laten")
        print(korelasi.round(3).to_string())

    return {"fit": fit, "measurement": tabel,
            "discriminant": diskriminan, "latent_corr": korelasi}


if __name__ == "__main__":
    import os
    if os.path.exists(config.OUTPUT_FILENAME):
        df = pd.read_csv(config.OUTPUT_FILENAME)
    else:
        from latent import LatentGenerator
        from structural import StructuralModel
        from indicator import IndicatorGenerator
        from ordinal_engine import OrdinalEngine
        from human_response import HumanResponseInjector
        full = StructuralModel().apply(LatentGenerator().sample())
        cont, _ = IndicatorGenerator(full).generate_all()
        inj = HumanResponseInjector()
        df = inj.apply_ordinal(OrdinalEngine().convert_dataframe(
            inj.apply_continuous(cont)))
    report(df)
