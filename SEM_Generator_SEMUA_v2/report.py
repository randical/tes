"""
report.py
---------
Menulis laporan teks yang bisa ditelusuri balik.

Prinsip yang dipegang, tiap angka di laporan harus jelas berasal dari
LAPIS MANA. Angka lapis satu bersifat pendekatan dan bias ke atas.
Angka lapis dua berasal dari CFA sungguhan. Mencampur keduanya dalam
satu tabel tanpa penanda adalah cara paling mudah membuat kesalahan
yang sulit dilacak di kemudian hari.
"""

import os

import numpy as np

import config


class ReportGenerator:

    def __init__(self, hasil_optimizer, df_final):
        self.h = hasil_optimizer
        self.df = df_final

    def write(self, path=None):
        path = path or os.path.join(config.OUTPUT_DIR, "laporan_qc.txt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        baris = []
        t = baris.append

        t("=" * 74)
        t("LAPORAN MUTU DATASET SINTETIS")
        t("=" * 74)
        t("")
        t("PERINGATAN ETIKA")
        t("Dataset ini SINTETIS, dibuat untuk melatih alur analisis.")
        t("Angka di dalamnya tidak berasal dari responden mana pun dan tidak")
        t("boleh dilaporkan sebagai temuan penelitian.")
        t("")
        t(f"Seed dasar        : {self.h['seed']}")
        t(f"Percobaan ke      : {self.h['attempts']}")
        t(f"Jumlah responden  : {config.N_RESPONDENTS}")
        t(f"Titik skala       : {config.SCALE_POINTS}")
        t(f"Preset realisme   : {config.REALISM_PRESET}")
        t("")

        t("-" * 74)
        t("MODEL YANG DIRANCANG")
        t("-" * 74)
        for outcome, d in self.h["model"].diagnostics.items():
            inter = (" + " + " + ".join(d["suku_interaksi"])) if d["suku_interaksi"] else ""
            t(f"  {outcome} <- {' + '.join(d['prediktor'])}{inter}")
            t(f"      beta rancangan {d['beta_rancangan']}")
            t(f"      R2 rancangan   {d['R2_rancangan']}")
        if getattr(self.h["model"], "residual_corr", None):
            t(f"  korelasi residual {self.h['model'].residual_corr}")
        t("")

        gen = self.h["generator"]
        t(f"  muatan silang               : {len(gen.crossloadings)} item")
        t(f"  pasangan error berkorelasi  : {len(gen.correlated_error_pairs)}")
        t(f"  injeksi gaya jawab          : {self.h['injector'].log}")
        t("")

        t("-" * 74)
        t("LAPIS SATU, saringan cepat berbasis komponen utama")
        t("CATATAN, estimasi ini BIAS KE ATAS. Jangan dikutip sebagai hasil.")
        t("-" * 74)
        v = self.h["qc"]["validity"]["per_construct"]
        t(f"{'Konstruk':<10}{'CR':>8}{'AVE':>8}{'AVE-margin':>12}{'status':>9}")
        for k, d in v.items():
            t(f"{k:<10}{d['CR']:>8.3f}{d['AVE']:>8.3f}"
              f"{d['AVE_konservatif']:>12.3f}{'OK' if d['passed'] else 'GAGAL':>9}")
        t("")
        rel = self.h["qc"]["reliability"]["per_construct"]
        t("  Cronbach's Alpha " + str({k: d["alpha"] for k, d in rel.items()}))
        t("  distribusi " + str(self.h["qc"]["distribution"]))
        t("  duplikat " + str(self.h["qc"]["duplicates"]))
        t("")

        if self.h.get("cfa"):
            c = self.h["cfa"]
            t("-" * 74)
            t("LAPIS DUA, CFA sungguhan. INI angka yang sah dikutip.")
            t("-" * 74)
            fit = c["fit"]
            t(f"  CMIN/DF {fit['chi2'] / fit['DoF']:.3f}   CFI {fit['CFI']:.3f}   "
              f"TLI {fit['TLI']:.3f}   RMSEA {fit['RMSEA']:.3f}")
            t(f"  GFI {fit['GFI']:.3f}   AGFI {fit['AGFI']:.3f}   NFI {fit['NFI']:.3f}")
            t("")
            t(f"{'Konstruk':<10}{'load min':>10}{'load rata':>11}{'CR':>8}{'AVE':>8}")
            for k, d in c["measurement"].items():
                t(f"{k:<10}{d['loading_min']:>10.3f}{d['loading_mean']:>11.3f}"
                  f"{d['CR']:>8.3f}{d['AVE']:>8.3f}")
            t("")
            t("  Validitas diskriminan Fornell-Larcker, korelasi LATEN")
            for k, d in c["discriminant"].items():
                t(f"    {k:<6} akar AVE {d['akar_AVE']:.3f} vs {d['korelasi_tertinggi']:.3f} "
                  f"({d['lawan']}) -> {'OK' if d['passed'] else 'GAGAL'}")
            t("")
            t("  Selisih lapis satu dikurangi lapis dua pada AVE")
            for k in c["measurement"]:
                selisih = v[k]["AVE"] - c["measurement"][k]["AVE"]
                t(f"    {k:<6}{selisih:>+8.3f}")
        t("")
        t("-" * 74)
        t(f"Kolom dataset akhir: {len(self.df.columns)}")
        t(f"Baris duplikat     : {int(self.df[config.all_items()].duplicated().sum())}")
        t("=" * 74)

        isi = "\n".join(baris)
        with open(path, "w", encoding="utf-8") as f:
            f.write(isi)
        return path, isi


if __name__ == "__main__":
    print("report.py tidak dijalankan sendiri. Pakai python main.py")
