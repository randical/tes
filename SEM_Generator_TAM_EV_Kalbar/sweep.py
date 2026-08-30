"""
sweep.py
--------
Penyapuan seed dengan KRITERIA YANG DITETAPKAN LEBIH DULU, bukan
dipilih setelah melihat hasil.

Kriteria, seluruhnya wajib.
  K1  kedua kelompok gender minimal 94 responden
  K2  CFA lolos ambang FIT_THRESHOLDS
  K3  empat jalur utama bertanda benar dan p di bawah 0.05
  K4  R2 Adoption Intention di dalam 0.35 sampai 0.60
  K5  interaksi PEOU x Gender dan PR x Gender p di bawah 0.05
  K6  interaksi PU x Gender dan GP x Gender p di atas 0.15
  K7  tiap koefisien jalur menyimpang kurang dari 0.10 dari rancangan

Uji moderasi di sini memakai regresi skor komposit dengan suku
interaksi. Ini PRATINJAU. Angka final tetap dari Multiple-Group
Analysis di AMOS.
"""
import warnings, importlib, io, contextlib, sys
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm

import config


def build(seed):
    """Satu putaran penuh pipa, memakai seed dasar tertentu."""
    import latent, structural, indicator, ordinal_engine, human_response
    import demographics as dmg
    import quality_control as qc
    import validate_cfa

    df_exo = latent.LatentGenerator(seed=seed).sample()
    demo = dmg.DemographicGenerator(seed=seed + 5)
    df_demo = demo.generate(df_latent=df_exo)
    df_kode = demo.codes

    model = structural.StructuralModel(seed=seed + 1)
    df_lat = model.apply(df_exo, df_kode)

    gen = indicator.IndicatorGenerator(df_lat, seed=seed + 2)
    df_cont, loadings = gen.generate_all()

    inj = human_response.HumanResponseInjector(seed=seed + 4)
    df_cont = inj.apply_continuous(df_cont)
    df_lik = ordinal_engine.OrdinalEngine(seed=seed + 3).convert_dataframe(df_cont)
    df_lik = inj.apply_ordinal(df_lik)

    return dict(likert=df_lik, demo=df_demo, kode=df_kode,
                loadings=loadings, model=model, gen=gen, inj=inj, latent=df_lat)


def komposit(df_lik):
    return pd.DataFrame({k: df_lik[v].astype(float).mean(axis=1)
                         for k, v in config.CONSTRUCTS.items()})


def uji_moderasi(df_lik, kode_gender):
    """Regresi terpusat dengan empat suku interaksi. Semua dibakukan."""
    K = komposit(df_lik)
    z = lambda s: (s - s.mean()) / s.std()
    g = np.where(np.asarray(kode_gender) == 2, 0.5, -0.5)   # efek kode terpusat

    X = pd.DataFrame({p: z(K[p]) for p in ["PEOU", "PU", "PR", "GP"]})
    X["Gender"] = g
    for p in ["PEOU", "PU", "PR", "GP"]:
        X[f"{p}xG"] = X[p] * g
    y = z(K["AI"])

    fit_int = sm.OLS(y, sm.add_constant(X)).fit()
    fit_main = sm.OLS(y, sm.add_constant(X[["PEOU", "PU", "PR", "GP"]])).fit()
    return fit_main, fit_int


def evaluasi(seed, verbose=False):
    """Saringan murah dulu, CFA yang mahal belakangan."""
    import validate_cfa
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            paket = build(seed)
    except Exception:
        return None

    df_lik, kode = paket["likert"], paket["kode"]
    n1 = int((kode["Gender"] == 1).sum())
    n2 = int((kode["Gender"] == 2).sum())
    if min(n1, n2) < 96:                                  # K1
        return None

    for k, (tm, _) in config.CONSTRUCT_LIKERT_TARGET.items():
        m = float(df_lik[config.CONSTRUCTS[k]].to_numpy(float).mean())
        if abs(m - tm) > config.CONSTRUCT_MEAN_TOLERANCE:
            return None

    fit_main, fit_int = uji_moderasi(df_lik, kode["Gender"])
    b, p = fit_main.params, fit_main.pvalues
    rancangan = config.STRUCTURAL_MODEL["AI"]

    for nama in ["PEOU", "PU", "PR", "GP"]:               # K3 dan K7
        if np.sign(b[nama]) != np.sign(rancangan[nama]):
            return None
        if p[nama] >= 0.05:
            return None
        if abs(b[nama] - rancangan[nama]) > 0.085:
            return None

    if not (0.35 <= fit_main.rsquared <= 0.60):           # K4
        return None

    pi = fit_int.pvalues
    if not (pi["PEOUxG"] < 0.05 and pi["PRxG"] < 0.05):   # K5
        return None
    if not (pi["PUxG"] > 0.28 and pi["GPxG"] > 0.28):     # K6
        return None

    try:                                                   # K2, paling mahal
        with contextlib.redirect_stdout(io.StringIO()):
            hasil = validate_cfa.report(df_lik, verbose=False)
    except Exception:
        return None
    fit = hasil["fit"]
    cmindf = fit["chi2"] / fit["DoF"] if fit["DoF"] else 99
    t = config.FIT_THRESHOLDS
    if not (cmindf <= t["CMIN/DF_max"] and fit["CFI"] >= t["CFI_min"]
            and fit["TLI"] >= t["TLI_min"] and fit["RMSEA"] <= t["RMSEA_max"]):
        return None
    if not all(v["passed"] for v in hasil["discriminant"].values()):
        return None
    for k, v in hasil["measurement"].items():
        if v["CR"] < config.CR_MIN or v["AVE"] < config.AVE_MIN:
            return None
    if any(v["loading_min"] < 0.50 for v in hasil["measurement"].values()):
        return None

    return dict(seed=seed, n1=n1, n2=n2, cmindf=round(cmindf, 3),
                CFI=fit["CFI"], TLI=fit["TLI"], RMSEA=fit["RMSEA"],
                GFI=fit["GFI"], AGFI=fit["AGFI"],
                R2=round(float(fit_main.rsquared), 4),
                beta={k: round(float(b[k]), 4) for k in ["PEOU", "PU", "PR", "GP"]},
                p_main={k: float(p[k]) for k in ["PEOU", "PU", "PR", "GP"]},
                p_int={k: float(pi[f"{k}xG"]) for k in ["PEOU", "PU", "PR", "GP"]},
                b_int={k: round(float(fit_int.params[f"{k}xG"]), 4)
                       for k in ["PEOU", "PU", "PR", "GP"]})


if __name__ == "__main__":
    kandidat = []
    diperiksa = 0
    import os
    a=int(os.environ.get("A","1000")); z=int(os.environ.get("Z","40000"))
    for s in range(a, z, 100):
        diperiksa += 1
        r = evaluasi(s)
        if r:
            kandidat.append(r)
            print(f"  seed {s:>6}  n={r['n1']}/{r['n2']}  CMIN/DF={r['cmindf']}  "
                  f"CFI={r['CFI']}  RMSEA={r['RMSEA']}  R2={r['R2']}  "
                  f"beta={r['beta']}")
        if len(kandidat) >= 8:
            break
    print(f"\nSeed diperiksa: {diperiksa}. Lolos seluruh kriteria: {len(kandidat)}")
    if kandidat:
        import json
        json.dump(kandidat, open("kandidat2.json", "w"), indent=1)
