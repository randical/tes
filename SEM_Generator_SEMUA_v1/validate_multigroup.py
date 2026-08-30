"""
validate_multigroup.py
-----------------------
Alat pemeriksa TAMBAHAN: menguji peran moderasi Environmental Concern
(H7, H7a-H7f) lewat Multi-Group Analysis -- persis Tabel 6 & 7 di
dokumen konteks penelitian Anda.

Langkah:
1) Bagi 200 responden jadi kelompok Low EC vs High EC (median split
   dari skor komposit item EC1-EC5 -- pendekatan umum dipakai karena
   kita cuma punya data indikator, bukan skor laten aslinya).
2) Fit model struktural (CL ~ PE+EE+SI+FC+HM+PV) TERPISAH di tiap
   kelompok (model tidak dikekang/unconstrained).
3) Fit model yang SAMA di seluruh data sekaligus, tanpa membedakan
   kelompok (model dikekang -- mengasumsikan jalurnya sama untuk
   semua orang).
4) Bandingkan chi-square kedua pendekatan (Nested Model Comparison):
   kalau model unconstrained jauh lebih baik (selisih chi-square
   besar & signifikan), berarti jalurnya MEMANG berbeda antar
   kelompok -- itulah bukti statistik moderasi H7.

Cara pakai: jalankan main.py dulu, baru:
    python validate_multigroup.py
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from semopy.multigroup import multigroup

import config


def moderated_regression_test(df):
    """
    KENAPA:
    Cara standar (dan lebih sederhana) untuk menguji moderasi secara
    statistik: masukkan suku interaksi (prediktor x EC) ke dalam satu
    regresi. Kalau koefisien suku interaksi itu signifikan (p<0.05),
    berarti kekuatan pengaruh prediktor itu terhadap CL MEMANG
    berubah sesuai level EC -- itulah definisi statistik moderasi.
    Ini alternatif yang lebih andal dibanding membandingkan
    chi-square multi-group yang constrained-nya sulit dipasang benar
    di semopy.

    BAGAIMANA:
    1) Hitung skor komposit (rata-rata item) tiap konstruk -- karena
       regresi interaksi butuh angka tunggal per konstruk, bukan
       banyak indikator.
    2) "Center" semua variabel (kurangi rata-ratanya) -- praktik
       standar supaya prediktor utama tidak terlalu berkorelasi
       dengan suku interaksinya sendiri.
    3) Buat suku interaksi: PE_x_EC = PE (sudah di-center) dikali EC
       (sudah di-center), begitu juga untuk 5 prediktor lainnya.
    4) Regresi OLS: CL ~ 6 prediktor + EC + 6 suku interaksi.
    """
    composite = pd.DataFrame({
        c: df[items].mean(axis=1) for c, items in config.CONSTRUCTS.items()
    })
    centered = composite - composite.mean()

    predictors = config.STRUCTURAL_PREDICTORS
    moderator = config.MODERATOR_CONSTRUCT
    outcome = config.STRUCTURAL_OUTCOME

    X = centered[predictors + [moderator]].copy()
    for p in predictors:
        X[f"{p}_x_{moderator}"] = centered[p] * centered[moderator]
    X = sm.add_constant(X)
    y = centered[outcome]

    return sm.OLS(y, X).fit()



def build_structural_desc():
    """Model pengukuran (semua 8 konstruk) + satu baris struktural H1-H6."""
    lines = [f"{c} =~ " + " + ".join(items) for c, items in config.CONSTRUCTS.items()]
    predictors = " + ".join(config.STRUCTURAL_PREDICTORS)
    lines.append(f"{config.STRUCTURAL_OUTCOME} ~ {predictors}")
    return "\n".join(lines)


def significance_stars(p_value):
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    return ""


if __name__ == "__main__":
    df = pd.read_csv(config.OUTPUT_FILENAME)

    # KENAPA: kita cuma punya data indikator (EC1-EC5), bukan skor
    # laten EC aslinya -- skor komposit (rata-rata item) adalah
    # proksi standar yang dipakai untuk membagi kelompok Low/High.
    ec_items = config.CONSTRUCTS[config.MODERATOR_CONSTRUCT]
    ec_composite = df[ec_items].mean(axis=1)
    median_ec = ec_composite.median()
    df["EC_GROUP"] = np.where(ec_composite >= median_ec, "High", "Low")

    print("=== VALIDASI MULTI-GROUP: EC SEBAGAI MODERATOR (H7) ===")
    print(f"Jumlah Low EC : {(df['EC_GROUP'] == 'Low').sum()}")
    print(f"Jumlah High EC: {(df['EC_GROUP'] == 'High').sum()}")

    desc = build_structural_desc()

    # --- Deskriptif: koefisien jalur per kelompok (gaya Tabel 7) ---
    result = multigroup(desc, df, group="EC_GROUP")

    # --- Uji statistik moderasi: regresi dengan suku interaksi ---
    reg = moderated_regression_test(df)

    print("\n--- Uji signifikansi moderasi (regresi suku interaksi) ---")
    print(f"{'Interaksi':<15}{'Koefisien':>12}{'p-value':>10}  Sig")
    interaksi_signifikan = []
    for predictor in config.STRUCTURAL_PREDICTORS:
        term = f"{predictor}_x_{config.MODERATOR_CONSTRUCT}"
        coef = reg.params[term]
        p = reg.pvalues[term]
        stars = significance_stars(p)
        if p < 0.05:
            interaksi_signifikan.append(predictor)
        print(f"{term:<15}{coef:>12.3f}{p:>10.4f}  {stars}")

    print(f"\nR^2 model (termasuk interaksi): {reg.rsquared:.3f}")
    print(f"\nJumlah jalur dengan bukti moderasi signifikan (p<0.05): "
          f"{len(interaksi_signifikan)} dari {len(config.STRUCTURAL_PREDICTORS)}")
    if interaksi_signifikan:
        print(f"Jalur yang terbukti dimoderasi EC: {', '.join(interaksi_signifikan)}")

    # --- Tabel 7 ala dokumen: koefisien jalur per kelompok ---
    print("\n--- Tabel 7 ala dokumen: Path coefficients per kelompok ---")
    print(f"{'Jalur':<10}{'Low EC':>12}{'High EC':>12}  {'Efek moderasi':<15}Sesuai H7?")

    low_paths = result.estimates["Low"]
    high_paths = result.estimates["High"]
    outcome = config.STRUCTURAL_OUTCOME

    for predictor in config.STRUCTURAL_PREDICTORS:
        row_low = low_paths[(low_paths["lval"] == outcome) & (low_paths["rval"] == predictor)]
        row_high = high_paths[(high_paths["lval"] == outcome) & (high_paths["rval"] == predictor)]

        est_low = row_low["Estimate"].values[0]
        est_high = row_high["Estimate"].values[0]
        p_low = row_low["p-value"].values[0]
        p_high = row_high["p-value"].values[0]

        arah_aktual = "Memperkuat" if est_high > est_low else "Memperlemah"
        arah_hipotesis = ("Memperkuat"
                           if config.STRUCTURAL_PATHS_HIGH_EC[predictor] > config.STRUCTURAL_PATHS_LOW_EC[predictor]
                           else "Memperlemah")
        cocok = "Terbukti" if arah_aktual == arah_hipotesis else "Beda arah"

        label_low = f"{est_low:.3f}{significance_stars(float(p_low)) if isinstance(p_low, (int, float)) else ''}"
        label_high = f"{est_high:.3f}{significance_stars(float(p_high)) if isinstance(p_high, (int, float)) else ''}"
        print(f"{predictor+'->'+outcome:<10}{label_low:>12}{label_high:>12}  {arah_aktual:<15}{cocok}")

    print("\n(* p<0.05, ** p<0.01, *** p<0.001)")
    print("\nCatatan metodologis: tabel di atas (path per kelompok) bersifat DESKRIPTIF --")
    print("menunjukkan arah & besar bedanya. Uji SIGNIFIKANSI moderasi yang sahih ada di")
    print("bagian 'regresi suku interaksi' di atas -- itu yang menentukan apakah H7a-H7f")
    print("terbukti secara statistik, bukan sekadar beda angka antar kelompok.")
