"""
main.py
-------
Titik masuk utama proyek. Menjalankan seluruh pipeline generator
dataset SEM dari awal sampai akhir:

    optimizer.py (di dalamnya: latent -> indicator -> ordinal ->
    quality_control, diulang sampai lolos) -> export.py

Cara pakai: buka terminal di folder proyek, ketik
    python main.py
"""

from optimizer import DatasetOptimizer
from export import DatasetExporter
from report import ReportGenerator

import config


def main():
    """
    KENAPA:
    Logika "jalankan semuanya" dibungkus jadi satu function, bukan
    ditulis lepas di bawah if __name__. Dengan begitu, kalau nanti
    Anda ingin memanggil pipeline ini dari tempat lain (Colab, GUI,
    script pengujian), tinggal `import main; main.main()` -- tanpa
    perlu menyalin ulang kodenya.

    BAGAIMANA:
    Panggil optimizer sampai dapat dataset yang lolos (atau mentok di
    batas percobaan), lalu simpan hasilnya lewat export.py, sambil
    mencetak ringkasan supaya Anda tahu apa yang terjadi tanpa perlu
    baca kode.
    """
    total_indikator = 0
    for daftar_item in config.CONSTRUCTS.values():
        total_indikator += len(daftar_item)

    print("=== GENERATOR DATASET SEM ===")
    print(f"Target: {config.N_RESPONDENTS} responden x {total_indikator} indikator\n")

    optimizer = DatasetOptimizer()
    df_final, loadings_final, report_final, n_attempts = optimizer.run()

    if not report_final["overall_passed"]:
        print(f"\nPeringatan: dataset TIDAK lolos semua syarat QC setelah "
              f"{n_attempts} percobaan, tapi tetap disimpan supaya bisa diperiksa.")

    exporter = DatasetExporter()
    saved_path = exporter.export(df_final)

    reporter = ReportGenerator(df_final, report_final)
    report_path = reporter.write_text_report()
    heatmap_path = reporter.save_heatmap()

    print("\n=== SELESAI ===")
    print(f"Dataset disimpan di : {saved_path}")
    print(f"Laporan QC          : {report_path}")
    print(f"Heatmap korelasi    : {heatmap_path}")
    print(f"Ukuran akhir        : {df_final.shape[0]} baris x {df_final.shape[1]} kolom")
    print(f"Jumlah percobaan    : {n_attempts}")
    print(f"Status QC           : {'LOLOS' if report_final['overall_passed'] else 'GAGAL'}")


if __name__ == "__main__":
    main()
