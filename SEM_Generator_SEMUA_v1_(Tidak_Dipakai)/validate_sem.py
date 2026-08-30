"""
validate_sem.py
----------------
Alat pemeriksa TAMBAHAN: menguji dataset_sem.csv lewat model SEM
PENUH (model pengukuran + model struktural sekaligus) memakai
semopy -- menghasilkan koefisien jalur dan signifikansinya, persis
seperti output "Regression Weights" di AMOS atau "Path Coefficients"
di SmartPLS.

Beda dengan validate_cfa.py: validate_cfa.py hanya menguji model
PENGUKURAN (apakah item-item valid mengukur konstruknya). File ini
menambahkan model STRUKTURAL (apakah PE, EE, dst berpengaruh
signifikan terhadap CL) -- dua pertanyaan yang berbeda dalam SEM.

Cara pakai: jalankan main.py dulu, baru:
    python validate_sem.py
"""

import pandas as pd
import semopy
from semopy import calc_stats

import config


def build_model_syntax():
    """
    KENAPA:
    Model SEM penuh butuh dua jenis baris: baris pengukuran ("=~",
    "diukur oleh") untuk tiap konstruk, DAN baris struktural ("~",
    "dipengaruhi oleh") untuk konstruk endogen -- dua notasi berbeda
    karena dua pertanyaan yang berbeda.

    BAGAIMANA:
    Baris pengukuran sama seperti di validate_cfa.py. Baris struktural
    ditambahkan satu baris di akhir: "CL ~ PE + EE + SI + ...".
    """
    lines = []
    for construct, items in config.CONSTRUCTS.items():
        indicators = " + ".join(items)
        lines.append(f"{construct} =~ {indicators}")

    predictors = " + ".join(config.STRUCTURAL_PREDICTORS)
    lines.append(f"{config.STRUCTURAL_OUTCOME} ~ {predictors}")

    return "\n".join(lines)


def significance_stars(p_value):
    """Bantuan kecil, notasi bintang yang biasa dipakai di tabel jurnal."""
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    return ""


if __name__ == "__main__":
    print("=== VALIDASI SEM PENUH (semopy) ===")
    print("\nModel yang diuji:")
    print(build_model_syntax())

    df = pd.read_csv(config.OUTPUT_FILENAME)
    model = semopy.Model(build_model_syntax())
    model.fit(df)

    estimates = model.inspect()
    outcome = config.STRUCTURAL_OUTCOME

    # Baris jalur struktural: lval == outcome DAN op == "~"
    # (baris pengukuran punya op yang sama, tapi lval-nya nama item, bukan CL)
    structural_rows = estimates[(estimates["lval"] == outcome) & (estimates["op"] == "~")]

    print(f"\n--- Koefisien jalur ke {outcome} ---")
    print(f"{'Prediktor':<10}{'Estimate':>10}{'Std.Err':>10}{'z-value':>10}{'p-value':>10}  Sig")
    for _, row in structural_rows.iterrows():
        p = row["p-value"]
        p_display = p if isinstance(p, float) else float(p)
        stars = significance_stars(p_display)
        print(f"{row['rval']:<10}{row['Estimate']:>10.3f}{row['Std. Err']:>10.3f}"
              f"{row['z-value']:>10.3f}{p_display:>10.4f}  {stars}")

    print("\n(* p<0.05, ** p<0.01, *** p<0.001 -- notasi standar tabel jurnal)")

    stats = calc_stats(model)
    s = stats.loc["Value"]
    print("\n--- Fit index model keseluruhan ---")
    print(f"CFI  : {s['CFI']:.3f}")
    print(f"TLI  : {s['TLI']:.3f}")
    print(f"RMSEA: {s['RMSEA']:.3f}")

    print(f"\nCatatan: koefisien jalur di atas adalah efek gabungan Low+High EC")
    print(f"(model ini belum memisah kelompok EC -- itu tugas validate_multigroup.py).")
    print(f"Wajar sedikit berbeda dari 'nilai asli' di config.py, sama seperti")
    print(f"membandingkan populasi vs sampel.")
