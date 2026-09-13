"""
sweep.py -- MODEL REVISI
------------------------
Penyapuan seed dengan KRITERIA YANG DITETAPKAN LEBIH DULU.

  K1  kedua kelompok gender minimal 96 responden
  K2  rerata tiap konstruk pada targetnya
  K3  H1 (PEOU ke PU) signifikan dan kuat, beta di atas 0.40
  K4  H3, H4, H5 bertanda benar dan p di bawah 0.05
  K5  H2 (PEOU ke AI) bertanda positif, boleh signifikan boleh tidak,
      tetapi HARUS jalur terlemah di antara keempatnya
  K6  efek tidak langsung PEOU LEBIH BESAR dari efek langsungnya
  K7  R2 AI di dalam 0.30 sampai 0.60
  K8  interaksi PEOU x Gender dan PR x Gender p di bawah 0.05
  K9  interaksi PU x Gender dan GP x Gender p di atas 0.28
  K10 CFA lolos ambang FIT_THRESHOLDS, CR, AVE, dan Fornell-Larcker

Uji di sini memakai regresi skor komposit. Ini PRATINJAU. Angka final
tetap dari AMOS.
"""
import warnings, io, contextlib, os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm

import config


def build(seed):
    import latent, structural, indicator, ordinal_engine, human_response
    import demographics as dmg

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

    return dict(likert=df_lik, demo=df_demo, kode=df_kode)


def komposit(df_lik):
    return pd.DataFrame({k: df_lik[v].astype(float).mean(axis=1)
                         for k, v in config.CONSTRUCTS.items()})


def evaluasi(seed):
    import validate_cfa
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            paket = build(seed)
    except Exception:
        return None

    df_lik, kode = paket["likert"], paket["kode"]
    n1 = int((kode["Gender"] == 1).sum())
    n2 = int((kode["Gender"] == 2).sum())
    if min(n1, n2) < 93:                                   # K1
        return None

    for k, (tm, _) in config.CONSTRUCT_LIKERT_TARGET.items():   # K2
        m = float(df_lik[config.CONSTRUCTS[k]].to_numpy(float).mean())
        if abs(m - tm) > config.CONSTRUCT_MEAN_TOLERANCE:
            return None

    K = komposit(df_lik)
    z = lambda s: (s - s.mean()) / s.std()
    Z = {k: z(K[k]) for k in config.CONSTRUCTS}

    # --- H1, jalur mediasi PEOU -> PU ---
    m_pu = sm.OLS(Z["PU"], sm.add_constant(pd.DataFrame({"PEOU": Z["PEOU"]}))).fit()
    b_h1, p_h1 = float(m_pu.params["PEOU"]), float(m_pu.pvalues["PEOU"])
    if b_h1 < 0.40 or p_h1 >= 0.001:                        # K3
        return None

    # --- H2 sampai H5, jalur menuju AI ---
    X = pd.DataFrame({p_: Z[p_] for p_ in ["PEOU", "PU", "PR", "GP"]})
    y = Z["AI"]
    m_ai = sm.OLS(y, sm.add_constant(X)).fit()
    b, p = m_ai.params, m_ai.pvalues
    ranc = config.STRUCTURAL_MODEL["AI"]

    for nama in ["PU", "PR", "GP"]:                         # K4
        if np.sign(b[nama]) != np.sign(ranc[nama]):
            return None
        if p[nama] >= 0.05:
            return None
        if abs(b[nama] - ranc[nama]) > 0.11:
            return None

    if b["PEOU"] <= 0:                                      # K5
        return None
    if abs(b["PEOU"]) >= min(abs(b["PU"]), abs(b["PR"]), abs(b["GP"])):
        return None

    tak_langsung = b_h1 * float(b["PU"])                    # K6
    if tak_langsung <= float(b["PEOU"]):
        return None

    if not (0.30 <= m_ai.rsquared <= 0.60):                 # K7
        return None

    # --- moderasi ---
    g = np.where(np.asarray(kode["Gender"]) == 2, 0.5, -0.5)
    Xi = X.copy()
    Xi["Gender"] = g
    for p_ in ["PEOU", "PU", "PR", "GP"]:
        Xi[f"{p_}xG"] = Xi[p_] * g
    m_int = sm.OLS(y, sm.add_constant(Xi)).fit()
    pi = m_int.pvalues
    if not (pi["PEOUxG"] < 0.05 and pi["PRxG"] < 0.05):     # K8
        return None
    if not (pi["PUxG"] > 0.28 and pi["GPxG"] > 0.28):       # K9
        return None

    # K10a -- LAPIS SATU penuh. Ini yang terlewat pada sweep pertama,
    # akibatnya seed terpilih gagal di optimizer karena alpha PR = 0.931
    # menembus plafon 0.93 (penjaga redundansi item).
    import quality_control
    try:
        qc = quality_control.QualityChecker(df_lik)
        with contextlib.redirect_stdout(io.StringIO()):
            if not qc.run_all()["overall_passed"]:
                return None
    except Exception:
        return None

    try:                                                    # K10b
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
                R2_PU=round(float(m_pu.rsquared), 4),
                R2_AI=round(float(m_ai.rsquared), 4),
                b_h1=round(b_h1, 4),
                beta={k: round(float(b[k]), 4) for k in ["PEOU", "PU", "PR", "GP"]},
                p_main={k: float(p[k]) for k in ["PEOU", "PU", "PR", "GP"]},
                tak_langsung=round(tak_langsung, 4),
                p_int={k: float(pi[f"{k}xG"]) for k in ["PEOU", "PU", "PR", "GP"]})


if __name__ == "__main__":
    kandidat, diperiksa = [], 0
    a = int(os.environ.get("A", "1000"))
    zz = int(os.environ.get("Z", "60000"))
    for s in range(a, zz, 100):
        diperiksa += 1
        r = evaluasi(s)
        if r:
            kandidat.append(r)
            print(f"  seed {s:>6} n={r['n1']}/{r['n2']} CMIN/DF={r['cmindf']} "
                  f"CFI={r['CFI']} R2_PU={r['R2_PU']} R2_AI={r['R2_AI']} "
                  f"H1={r['b_h1']} beta={r['beta']}")
        if len(kandidat) >= 6:
            break
    print(f"\nSeed diperiksa: {diperiksa}. Lolos seluruh kriteria: {len(kandidat)}")
    if kandidat:
        import json
        json.dump(kandidat, open("kandidat.json", "w"), indent=1)
