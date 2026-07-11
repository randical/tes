"""
validate_cfa.py
----------------
Alat pemeriksa TAMBAHAN (bukan bagian dari pipeline generator).
Menguji dataset_sem.csv yang sudah dihasilkan lewat CFA sungguhan
memakai semopy, untuk melihat fit index (CFI, TLI, RMSEA) -- ukuran
yang TIDAK dicek oleh quality_control.py karena nilainya bergantung
pada model SEM yang diuji, bukan cuma pada datanya sendiri.

Cara pakai: pastikan sudah menjalankan main.py dulu (supaya
output/dataset_sem.csv ada), lalu:
    python validate_cfa.py
"""

import pandas as pd
import semopy
from semopy import calc_stats

import config


def build_model_syntax():
    """
    KENAPA:
    semopy (sama seperti lavaan di R) butuh "resep" model dalam
    bentuk teks, bukan dictionary Python biasa -- ini notasi standar
    di dunia software SEM, supaya bisa dibaca manusia maupun program.

    BAGAIMANA:
    Untuk tiap konstruk, tulis satu baris:
        NAMA_KONSTRUK =~ item1 + item2 + item3 + ...
    Tanda "=~" artinya "diukur oleh" -- konstruk laten di kiri
    diasumsikan menjadi penyebab dari indikator-indikator di kanan.
    Ini persis rumus yang kita pakai waktu MEMBUAT data di
    indicator.py, cuma sekarang arahnya dibalik: sebelumnya kita
    tahu loading lalu membuat data, sekarang semopy diminta MENEBAK
    loadingnya dari data.
    """
    lines = []
    for construct, items in config.CONSTRUCTS.items():
        indicators = " + ".join(items)
        lines.append(f"{construct} =~ {indicators}")
    return "\n".join(lines)


def run_cfa(csv_path):
    """
    KENAPA:
    Ini yang benar-benar menjalankan estimasi CFA -- proses ini yang
    di dunia nyata dilakukan AMOS/SmartPLS/lavaan.

    BAGAIMANA:
    Baca CSV, kasih tahu semopy modelnya lewat build_model_syntax(),
    lalu model.fit(df) mencari nilai loading & korelasi antar
    konstruk yang paling cocok dengan data. calc_stats() menghitung
    fit index dari hasil estimasi tadi.
    """
    df = pd.read_csv(csv_path)
    model_syntax = build_model_syntax()

    model = semopy.Model(model_syntax)
    model.fit(df)

    stats = calc_stats(model)
    return stats


def interpret(value, good, acceptable, higher_is_better=True):
    """Bantuan kecil untuk memberi label OK/LAYAK/KURANG pada tiap fit index."""
    if higher_is_better:
        if value >= good:
            return "SANGAT BAIK"
        elif value >= acceptable:
            return "LAYAK"
        return "KURANG"
    else:
        if value <= good:
            return "SANGAT BAIK"
        elif value <= acceptable:
            return "LAYAK"
        return "KURANG"


if __name__ == "__main__":
    print("=== VALIDASI CFA (semopy) ===")
    print("\nModel yang diuji:")
    print(build_model_syntax())

    stats = run_cfa(config.OUTPUT_FILENAME)
    s = stats.loc["Value"]  # ambil baris "Value" saja, isinya semua angka fit index

    print("\n--- Fit index ---")
    print(f"Chi-square       : {s['chi2']:.2f} (df={s['DoF']:.0f}, p={s['chi2 p-value']:.4f})")
    print(f"CFI              : {s['CFI']:.3f}  -> {interpret(s['CFI'], 0.95, 0.90)}")
    print(f"TLI              : {s['TLI']:.3f}  -> {interpret(s['TLI'], 0.95, 0.90)}")
    print(f"RMSEA            : {s['RMSEA']:.3f}  -> "
          f"{interpret(s['RMSEA'], 0.06, 0.08, higher_is_better=False)}")

    print("\nPatokan umum: CFI/TLI >= 0.90 layak, >= 0.95 sangat baik.")
    print("RMSEA <= 0.08 layak, <= 0.06 sangat baik. (Hu & Bentler, 1999)")
    print("\nCatatan: hasil ini TIDAK dicek otomatis oleh quality_control.py")
    print("karena bergantung pada model SEM, bukan cuma data mentahnya.")
    print("Alpha/CR/AVE tinggi tidak otomatis menjamin fit index bagus --")
    print("itu sebabnya validasi CFA terpisah ini tetap penting dijalankan.")
