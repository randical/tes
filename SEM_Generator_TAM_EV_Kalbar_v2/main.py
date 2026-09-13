"""
main.py
-------
Titik masuk. Jalankan dengan

    python main.py

Alurnya

    optimizer.py  (latent -> structural -> indicator -> gaya jawab ->
                   ordinal -> lapis 1 -> lapis 2, diulang sampai lolos)
        -> demographics.py
        -> product_indicators.py
        -> export.py
        -> report.py
"""

import warnings

warnings.filterwarnings("ignore")

import config
from optimizer import DatasetOptimizer
from demographics import DemographicGenerator
from product_indicators import build_all
from export import DatasetExporter
from report import ReportGenerator


def main():
    print("=" * 74)
    print("GENERATOR DATASET SEM SINTETIS")
    print("=" * 74)
    print("PERINGATAN, dataset ini sintetis. Untuk latihan alur analisis,")
    print("bukan pengganti data responden dan tidak boleh dilaporkan sebagai")
    print("temuan penelitian.\n")
    print(f"Model    : {len(config.CONSTRUCTS)} konstruk, "
          f"{len(config.all_items())} indikator, n={config.N_RESPONDENTS}")
    print(f"Endogen  : {config.endogenous_constructs()}")
    print(f"Eksogen  : {config.exogenous_constructs()}")
    print(f"Moderasi : {[f'{m['predictor']}x{m['moderator']} -> {m['outcome']}' for m in config.MODERATIONS]}")
    print(f"Realisme : preset '{config.REALISM_PRESET}'\n")

    print("Mencari dataset yang lolos dua lapis")
    hasil = DatasetOptimizer().run()
    df_likert = hasil["df_likert"]

    # Demografi TIDAK dibangkitkan ulang di sini. Kalau dibangkitkan ulang
    # dengan seed yang sama tapi di luar loop, kolom kelompok dan kovariat
    # bisa tidak sinkron dengan yang benar-benar dipakai mesin struktural.
    # Optimizer sudah mengembalikan kolom demografi yang sesungguhnya dipakai.
    df_demo = hasil.get("demografi")

    df_produk, catatan_produk = build_all(df_likert, hasil["loadings"])

    df_final = DatasetExporter.assemble(df_likert, df_demo, df_produk,
                                        hasil.get("kode_demografi"))
    path = DatasetExporter.export(df_final)

    path_lap, isi = ReportGenerator(hasil, df_final).write()

    print("\n" + "=" * 74)
    print("SELESAI")
    print("=" * 74)
    print(f"Dataset  : {path}   ({df_final.shape[0]} baris x {df_final.shape[1]} kolom)")
    print(f"Laporan  : {path_lap}")
    if catatan_produk:
        for nama, pasangan in catatan_produk.items():
            print(f"Indikator produk {nama}: "
                  + ", ".join(f"{a}x{b}" for a, b in pasangan))
    print("\nLangkah berikutnya")
    print("  python validate_sem.py     pratinjau jalur, mediasi, moderasi")
    print("  lalu buka dataset di SPSS/AMOS mengikuti Modul 2")
    return df_final


if __name__ == "__main__":
    main()
