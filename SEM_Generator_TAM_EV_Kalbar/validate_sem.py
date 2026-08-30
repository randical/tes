"""
validate_sem.py
---------------
Menguji MODEL STRUKTURAL pada dataset yang sudah jadi.

Empat keluaran.

1. Koefisien jalur terstandardisasi beserta p.
2. Efek tidak langsung dengan selang kepercayaan bootstrap.
3. Uji moderasi cara UTAMA, interaksi laten lewat indikator produk.
4. Uji moderasi cara PENDAMPING, multi-grup belah median.

Angka dari modul ini adalah PRATINJAU, bukan angka final. Angka final
untuk naskah harus berasal dari AMOS. Gunanya di sini untuk memastikan
dataset memang mengandung pola yang Anda rancang, sebelum waktu Anda
terbuang menggambar model di AMOS.
"""

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import config

BOOTSTRAP_N = 400


# ==========================================================
# PENYUSUN SINTAKS
# ==========================================================
def build_sem_spec(with_interaction=False, constructs=None, model=None,
                   product_prefix=None):
    constructs = constructs or config.CONSTRUCTS
    model = model or config.STRUCTURAL_MODEL

    baris = [f"{k} =~ " + " + ".join(v) for k, v in constructs.items()]

    # Konstruk orde dua. Induk diukur oleh dimensinya, bukan oleh item.
    for induk, dimensi in getattr(config, "SECOND_ORDER", {}).items():
        baris.append(f"{induk} =~ " + " + ".join(dimensi))

    if with_interaction and product_prefix:
        for prefix, kolom in product_prefix.items():
            baris.append(f"{prefix} =~ " + " + ".join(kolom))

    for outcome, prediktor in model.items():
        daftar = list(prediktor)
        if with_interaction and product_prefix:
            for m in config.MODERATIONS:
                if m["outcome"] == outcome:
                    daftar.append(f'{m["predictor"]}x{m["moderator"]}')
        baris.append(f"{outcome} ~ " + " + ".join(daftar))

    for spec in getattr(config, "RESIDUAL_COVARIANCES", []):
        a, b = spec["between"]
        baris.append(f"{a} ~~ {b}")

    return "\n".join(baris)


def siapkan_kovariat(df):
    """
    Kovariat teramati dibaca dari kolom kode angka lalu dibakukan, persis
    seperti yang dilakukan mesin saat data dibangkitkan. Kalau tidak
    dibakukan dengan cara yang sama, koefisiennya tidak sebanding dengan
    angka rancangan di config.py.
    """
    keluar = {}
    for c in getattr(config, "COVARIATES", []):
        kol = f"{c}_kode"
        if kol not in df.columns:
            raise SystemExit(
                f"Kolom '{kol}' tidak ada di dataset. Pastikan main.py dijalankan "
                "ulang setelah COVARIATES diisi.")
        v = df[kol].astype(float)
        keluar[c] = (v - v.mean()) / v.std()
    return pd.DataFrame(keluar, index=df.index)


def test_moderation_group(df, df_kode, produk_cols=None):
    """
    Pratinjau moderasi oleh variabel KATEGORIK. Uji resminya tetap
    Multiple-Group Analysis di AMOS, lihat Modul 2 bagian 6.

    PERANGKAP YANG PERLU DIKETAHUI:
    Kalau jalur yang sama juga dimoderasi oleh konstruk kontinu lewat
    MODERATIONS, suku interaksinya WAJIB tetap ada di dalam model tiap
    kelompok. Kalau dihilangkan, koefisien per kelompok akan menyerap
    sebagian efek interaksi kontinu itu dan angkanya menyimpang jauh dari
    rancangan. Diukur di sandbox, penyimpangannya sampai 0,19 pada satu
    kelompok, cukup besar untuk membalik kesimpulan.
    """
    daftar = getattr(config, "GROUP_MODERATIONS", [])
    if not daftar:
        return
    print("\nMODERASI OLEH VARIABEL KATEGORIK (pratinjau, uji resmi di AMOS)")
    for g in daftar:
        kol = g["group_column"]
        kode = df_kode[f"{kol}_kode"].to_numpy()
        print(f"  Jalur {g['predictor']} -> {g['outcome']}, kelompok menurut {kol}")
        for k in sorted(set(kode)):
            sub = kode == k
            if sub.sum() < 50:
                print(f"    kelompok {k}  n={sub.sum():<4} terlalu kecil, dilewati")
                continue
            pakai_interaksi = bool(produk_cols)
            spec = build_sem_spec(with_interaction=pakai_interaksi,
                                  product_prefix=produk_cols)
            m = fit_sem(df[sub].astype(float), spec)
            t = path_table(m)
            baris = t[(t["lval"] == g["outcome"]) & (t["rval"] == g["predictor"])]
            beta = float(baris["beta_std"].iloc[0]) if len(baris) else float("nan")
            rancang = (config.STRUCTURAL_MODEL[g["outcome"]][g["predictor"]]
                       + float(g["deltas"].get(k, 0.0)))
            print(f"    kelompok {k}  n={sub.sum():<4} beta {beta:+.4f}"
                  f"   rancangan {rancang:+.3f}")
        print("    CATATAN, selisih antar kelompok di atas DESKRIPTIF. Uji formalnya")
        print("    adalah Delta chi-square per jalur, lihat Modul 2 bagian 6.5.")


def fit_sem(df, spec):
    from semopy import Model
    m = Model(spec)
    m.fit(df)
    return m


def path_table(model):
    ins = model.inspect(std_est=True)
    kolom = "Est. Std" if "Est. Std" in ins.columns else "Estimate"
    reg = ins[ins["op"] == "~"].copy()
    laten = set(config.CONSTRUCTS) | set(getattr(config, "SECOND_ORDER", {}))
    laten |= set(getattr(config, "COVARIATES", []))
    laten |= {f'{m["predictor"]}x{m["moderator"]}' for m in config.MODERATIONS}
    reg = reg[reg["lval"].isin(config.STRUCTURAL_MODEL) & reg["rval"].isin(laten)]
    return reg[["lval", "rval", kolom, "Est. Std" if kolom != "Est. Std" else "Estimate",
                "p-value"]].rename(columns={kolom: "beta_std"})


def _coef(model, outcome, predictor):
    ins = model.inspect(std_est=True)
    kolom = "Est. Std" if "Est. Std" in ins.columns else "Estimate"
    baris = ins[(ins["op"] == "~") & (ins["lval"] == outcome) & (ins["rval"] == predictor)]
    return float(baris[kolom].iloc[0]) if len(baris) else np.nan


# ==========================================================
# EFEK TIDAK LANGSUNG
# ==========================================================
def indirect_paths():
    """Menelusuri semua rantai X -> M -> ... -> Y dari STRUCTURAL_MODEL."""
    model = config.STRUCTURAL_MODEL
    rantai = []

    def telusuri(sekarang, jejak):
        for outcome, prediktor in model.items():
            if sekarang in prediktor and outcome not in jejak:
                baru = jejak + [outcome]
                if len(baru) >= 3:
                    rantai.append(list(baru))
                telusuri(outcome, baru)

    for c in config.CONSTRUCTS:
        telusuri(c, [c])
    return rantai


def indirect_effects(df, spec, rantai, n_boot=BOOTSTRAP_N, seed=99, verbose=True):
    rng = np.random.default_rng(seed)
    model = fit_sem(df, spec)

    titik = {}
    for r in rantai:
        nilai = 1.0
        for i in range(len(r) - 1):
            nilai *= _coef(model, r[i + 1], r[i])
        titik[" -> ".join(r)] = nilai

    simpanan = {k: [] for k in titik}
    n = len(df)
    berhasil = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            m = fit_sem(df.iloc[idx].reset_index(drop=True), spec)
        except Exception:
            continue
        ok = True
        sementara = {}
        for r in rantai:
            nilai = 1.0
            for i in range(len(r) - 1):
                c = _coef(m, r[i + 1], r[i])
                if np.isnan(c):
                    ok = False
                nilai *= c
            sementara[" -> ".join(r)] = nilai
        if ok:
            berhasil += 1
            for k, v in sementara.items():
                simpanan[k].append(v)

    hasil = {}
    for k, v in titik.items():
        arr = np.array(simpanan[k])
        if len(arr) < 30:
            hasil[k] = {"efek": round(v, 4), "CI": None, "signifikan": None}
            continue
        lo, hi = np.percentile(arr, [2.5, 97.5])
        hasil[k] = {"efek": round(v, 4),
                    "CI": (round(float(lo), 4), round(float(hi), 4)),
                    "signifikan": bool(lo > 0 or hi < 0)}

    if verbose:
        print(f"\nEFEK TIDAK LANGSUNG, bootstrap persentil {berhasil}/{n_boot} replikasi valid")
        print(f"{'Jalur':<28}{'efek':>9}{'CI bawah':>11}{'CI atas':>10}{'status':>16}")
        for k, v in hasil.items():
            if v["CI"]:
                s = "signifikan" if v["signifikan"] else "tidak signifikan"
                print(f"{k:<28}{v['efek']:>9.4f}{v['CI'][0]:>11.4f}"
                      f"{v['CI'][1]:>10.4f}{s:>16}")
            else:
                print(f"{k:<28}{v['efek']:>9.4f}{'-':>11}{'-':>10}{'gagal konvergen':>16}")
    return hasil


# ==========================================================
# MODERASI CARA UTAMA, INTERAKSI LATEN
# ==========================================================
def test_moderation_latent(df, product_cols, verbose=True):
    spec = build_sem_spec(with_interaction=True, product_prefix=product_cols)
    model = fit_sem(df, spec)
    ins = model.inspect(std_est=True)
    kolom = "Est. Std" if "Est. Std" in ins.columns else "Estimate"

    hasil = {}
    if verbose:
        print("\nMODERASI CARA UTAMA, interaksi laten dengan indikator produk")
        print(f"{'Suku interaksi':<22}{'beta std':>10}{'p':>10}{'status':>20}")
    for m in config.MODERATIONS:
        nama = f'{m["predictor"]}x{m["moderator"]}'
        baris = ins[(ins["op"] == "~") & (ins["lval"] == m["outcome"])
                    & (ins["rval"] == nama)]
        if len(baris) == 0:
            continue
        beta = float(baris[kolom].iloc[0])
        p = float(baris["p-value"].iloc[0])
        hasil[nama] = {"beta": round(beta, 4), "p": round(p, 4),
                       "rancangan": m["coefficient"],
                       "signifikan": p < 0.05}
        if verbose:
            s = "signifikan" if p < 0.05 else "tidak signifikan"
            print(f"{nama + ' -> ' + m['outcome']:<22}{beta:>10.4f}{p:>10.4f}{s:>20}")
            print(f"{'  (rancangan)':<22}{m['coefficient']:>10.4f}")
    return hasil


# ==========================================================
# MODERASI CARA PENDAMPING, MULTI-GRUP BELAH MEDIAN
# ==========================================================
def test_moderation_multigroup(df, verbose=True):
    hasil = {}
    if verbose:
        print("\nMODERASI CARA PENDAMPING, belah median (pratinjau, "
              "uji resmi tetap di AMOS)")
    for m in config.MODERATIONS:
        w_items = config.CONSTRUCTS[m["moderator"]]
        skor_w = df[w_items].astype(float).mean(axis=1)
        median = float(skor_w.median())
        rendah = df[skor_w <= median]
        tinggi = df[skor_w > median]

        spec = build_sem_spec()
        baris = {}
        for label, sub in [("rendah", rendah), ("tinggi", tinggi)]:
            try:
                mm = fit_sem(sub.reset_index(drop=True), spec)
                baris[label] = {
                    "n": len(sub),
                    "beta": round(_coef(mm, m["outcome"], m["predictor"]), 4),
                }
            except Exception as e:
                baris[label] = {"n": len(sub), "beta": None, "error": str(e)[:60]}

        hasil[f'{m["predictor"]}->{m["outcome"]} | {m["moderator"]}'] = {
            "median": round(median, 3), **baris}

        if verbose:
            print(f"  Jalur {m['predictor']} -> {m['outcome']}, "
                  f"moderator {m['moderator']} (median {median:.3f})")
            for label in ("rendah", "tinggi"):
                b = baris[label]
                nilai = b["beta"] if b["beta"] is not None else "gagal"
                print(f"    kelompok {label:<7} n={b['n']:<5} beta = {nilai}")
            if all(baris[l]["beta"] is not None for l in ("rendah", "tinggi")):
                selisih = baris["tinggi"]["beta"] - baris["rendah"]["beta"]
                arah = "melemahkan" if selisih < 0 else "menguatkan"
                print(f"    selisih tinggi dikurangi rendah = {selisih:+.4f} "
                      f"-> arah {arah}")
                print("    CATATAN, selisih ini DESKRIPTIF. Uji formalnya adalah "
                      "Delta chi-square")
                print("    baris Structural weights di AMOS Multiple-Group Analysis.")
    return hasil


if __name__ == "__main__":
    import os

    if os.path.exists(config.OUTPUT_FILENAME):
        df = pd.read_csv(config.OUTPUT_FILENAME)
    else:
        raise SystemExit("Jalankan python main.py lebih dulu.")

    items = config.all_items()
    produk_cols = {}
    for m in config.MODERATIONS:
        nama = f'{m["predictor"]}x{m["moderator"]}'
        kolom = [c for c in df.columns if c.startswith(nama + "_")]
        if kolom:
            produk_cols[nama] = kolom

    print("=== VALIDASI MODEL STRUKTURAL ===")
    kov = siapkan_kovariat(df)
    df_analisis = pd.concat([df[items].astype(float), kov], axis=1)

    spec = build_sem_spec()
    model = fit_sem(df_analisis, spec)

    print("\nKoefisien jalur terstandardisasi")
    tabel = path_table(model)
    print(f"{'Jalur':<18}{'beta std':>10}{'p':>10}{'rancangan':>12}")
    for _, r in tabel.iterrows():
        rancang = config.STRUCTURAL_MODEL.get(r["lval"], {}).get(r["rval"], "")
        rancang = f"{rancang:.3f}" if rancang != "" else "-"
        print(f"{r['rval'] + ' -> ' + r['lval']:<18}{r['beta_std']:>10.4f}"
              f"{r['p-value']:>10.4f}{rancang:>12}")

    rantai = indirect_paths()
    indirect_effects(df_analisis, spec, rantai)

    if produk_cols:
        kolom_produk = [c for v in produk_cols.values() for c in v]
        df_interaksi = pd.concat(
            [df[items + kolom_produk].astype(float), kov], axis=1)
        test_moderation_latent(df_interaksi, produk_cols)

    df_lengkap = df_analisis
    if produk_cols:
        df_lengkap = pd.concat(
            [df_analisis, df[[c for v in produk_cols.values() for c in v]]
             .astype(float)], axis=1)

    test_moderation_multigroup(df_lengkap)
    test_moderation_group(df_lengkap, df, produk_cols)
