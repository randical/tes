"""
export.py
---------
Menggabungkan seluruh blok kolom lalu menyimpannya.

Urutan kolom sengaja dibuat tetap, ID lalu demografi lalu indikator
lalu skor komposit lalu indikator produk lalu variabel pengelompokan.
Urutan yang stabil memudahkan Anda menyeret variabel di AMOS dan
memudahkan pembandingan antar putaran generate.
"""

import os

import numpy as np
import pandas as pd

import config


class DatasetExporter:

    @staticmethod
    def assemble(df_likert, df_demografi=None, df_produk=None, df_kode=None):
        blok = []

        n = len(df_likert)
        blok.append(pd.DataFrame({"ID": np.arange(1, n + 1)}, index=df_likert.index))

        if df_demografi is not None and len(df_demografi.columns):
            blok.append(df_demografi.set_index(df_likert.index))

        # Kolom kode angka untuk demografi yang dipakai sebagai kovariat
        # atau sebagai variabel pengelompokan. AMOS butuh angka, bukan teks.
        perlu_kode = set(getattr(config, "COVARIATES", []))
        perlu_kode |= {g["group_column"]
                       for g in getattr(config, "GROUP_MODERATIONS", [])}
        if df_kode is not None and perlu_kode:
            blok.append(df_kode[sorted(perlu_kode)].add_suffix("_kode")
                        .set_index(df_likert.index))

        blok.append(df_likert[config.all_items()])

        komposit = pd.DataFrame(
            {f"Mean_{k}": df_likert[v].astype(float).mean(axis=1).round(4)
             for k, v in config.CONSTRUCTS.items()},
            index=df_likert.index)
        blok.append(komposit)

        if df_produk is not None and len(df_produk.columns):
            blok.append(df_produk)

        # Variabel pengelompokan untuk multi-grup belah median di AMOS
        # Kolom kelompok untuk moderasi KATEGORIK sudah ada sebagai kolom
        # demografi asli, jadi tidak perlu dibuat ulang. Yang dibuat di bawah
        # hanya kolom belah median untuk moderator KONTINU.
        for m in config.MODERATIONS:
            kolom = f"Mean_{m['moderator']}"
            median = float(komposit[kolom].median())
            grup = np.where(komposit[kolom] <= median, 1, 2)
            blok.append(pd.DataFrame(
                {f"Grup_{m['moderator']}": grup}, index=df_likert.index))

        return pd.concat(blok, axis=1)

    @staticmethod
    def export(df, path=None):
        path = path or config.OUTPUT_FILENAME
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        return path


if __name__ == "__main__":
    print("export.py tidak dijalankan sendiri. Pakai python main.py")
