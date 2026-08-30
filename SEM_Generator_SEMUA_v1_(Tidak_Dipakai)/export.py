"""
export.py
---------
Menyimpan dataset final (yang sudah dinyatakan lolos oleh
optimizer.py) menjadi file CSV yang siap dipakai di SPSS, AMOS,
atau R/lavaan.
"""

import os
import pandas as pd

import config


class DatasetExporter:
    """
    Menangani penyimpanan dataset ke disk, termasuk membuat folder
    output kalau belum ada.
    """

    def __init__(self, filename=None):
        self.filename = filename if filename is not None else config.OUTPUT_FILENAME

    def export(self, df):
        """
        KENAPA:
        config.OUTPUT_FILENAME menunjuk ke folder "output/" yang
        mungkin belum ada di komputer Anda -- kalau langsung disimpan
        tanpa dicek, Python akan berhenti dengan error
        "No such file or directory".

        BAGAIMANA:
        os.makedirs(..., exist_ok=True) membuat foldernya otomatis
        tanpa menimbulkan error kalau ternyata sudah ada. df.to_csv()
        lalu menuliskan tabelnya. index=False dipakai supaya tidak
        ada kolom nomor baris tambahan yang tidak dibutuhkan.
        """
        folder = os.path.dirname(self.filename)
        if folder:
            os.makedirs(folder, exist_ok=True)

        df.to_csv(self.filename, index=False)
        return self.filename


# ==========================================================
# BLOK PENGECEKAN MANDIRI
# ==========================================================
# Untuk mengetes export.py sendirian, kita tidak perlu menjalankan
# optimizer.py (yang bisa mengulang berkali-kali) -- cukup satu kali
# generate saja, sekadar memastikan file tersimpan dengan benar.
# Disimpan ke nama file terpisah supaya tidak menimpa hasil asli
# dari main.py.
if __name__ == "__main__":
    from latent import LatentGenerator
    from structural import StructuralModel
    from indicator import IndicatorGenerator
    from ordinal_engine import OrdinalEngine

    df_exogenous = LatentGenerator().sample_latent()
    df_latent = StructuralModel().apply(df_exogenous)
    df_continuous, _ = IndicatorGenerator(df_latent).generate_all()
    df_likert = OrdinalEngine().convert_dataframe(df_continuous)

    exporter = DatasetExporter(filename="output/tes_export.csv")
    saved_path = exporter.export(df_likert)

    print("=== CEK EXPORT.PY ===")
    print(f"File tes tersimpan di: {saved_path}")
    print(f"Ukuran: {df_likert.shape[0]} baris x {df_likert.shape[1]} kolom")
