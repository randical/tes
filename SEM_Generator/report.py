"""
report.py
---------
Membuat laporan otomatis dari hasil quality_control.py: ringkasan
teks (lolos/gagal per pemeriksaan) dan heatmap korelasi 40x40 --
supaya setiap kali generate data baru, ada dokumentasi otomatis
tanpa perlu menyalin angka dari terminal secara manual.
"""

import os
import matplotlib.pyplot as plt

import config


class ReportGenerator:
    """
    Menangani penulisan laporan teks dan gambar heatmap ke folder
    output, berdasarkan hasil df_likert dan report dari
    quality_control.py.
    """

    def __init__(self, df_likert, report, output_dir="output"):
        self.df = df_likert
        self.report = report
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def write_text_report(self, filename="qc_report.txt"):
        """
        KENAPA:
        Supaya ada catatan permanen hasil pemeriksaan tiap kali data
        digenerate -- berguna kalau nanti Anda perlu menunjukkan bukti
        proses screening data ke pembimbing/reviewer jurnal.

        BAGAIMANA:
        "with open(path, 'w') as f:" adalah cara aman membuka file
        untuk ditulisi -- Python otomatis menutup file itu sendiri
        begitu blok selesai, bahkan kalau terjadi error di tengah
        jalan. f.write() menulis teksnya ke file.
        """
        path = os.path.join(self.output_dir, filename)

        lines = ["=== LAPORAN QUALITY CONTROL ==="]
        status = "LOLOS" if self.report["overall_passed"] else "GAGAL"
        lines.append(f"Status keseluruhan: {status}")

        lines.append(f"\nDuplikat baris: {self.report['duplicate']['n_duplicate']} "
                      f"(syarat maks {config.MAX_DUPLICATE_ROWS})")

        d = self.report["distribution"]
        lines.append(f"Rata-rata: {d['mean']:.3f} (target {config.TARGET_MEAN})")
        lines.append(f"SD       : {d['std']:.3f} (target {config.TARGET_STD})")

        lines.append("\nCronbach's Alpha per konstruk:")
        for construct, res in self.report["reliability"]["per_construct"].items():
            status_c = "OK" if res["passed"] else "MELESET"
            lines.append(f"  {construct}: {res['alpha']:.3f} -> {status_c}")

        lines.append("\nComposite Reliability & AVE per konstruk:")
        for construct, res in self.report["validity"]["per_construct"].items():
            status_c = "OK" if (res["cr_passed"] and res["ave_passed"]) else "MELESET"
            lines.append(f"  {construct}: CR={res['cr']:.3f}, AVE={res['ave']:.3f} -> {status_c}")

        lines.append("\nDiscriminant Validity (Fornell-Larcker):")
        for construct, res in self.report["discriminant"]["per_construct"].items():
            status_c = "OK" if res["passed"] else "MELESET"
            lines.append(f"  {construct}: sqrt(AVE)={res['sqrt_ave']:.3f} vs korelasi "
                          f"tertinggi={res['max_corr']:.3f} -> {status_c}")

        with open(path, "w") as f:
            f.write("\n".join(lines))

        return path

    def save_heatmap(self, filename="correlation_heatmap.png"):
        """
        KENAPA:
        Tabel angka korelasi 40x40 sulit dibaca sekilas; gambar
        heatmap membuat pola "item mana berkelompok dengan item mana"
        langsung terlihat -- kalau ada warna aneh nyasar di luar
        kelompoknya sendiri, itu tanda ada masalah (cross-loading).

        BAGAIMANA:
        matplotlib menggambar tiap sel korelasi sebagai kotak warna
        (merah = korelasi positif kuat, biru = negatif, lewat
        ax.imshow()), lalu disimpan sebagai file gambar PNG.
        """
        path = os.path.join(self.output_dir, filename)
        corr = self.df.corr()

        fig, ax = plt.subplots(figsize=(10, 9))
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
        ax.set_yticklabels(corr.columns, fontsize=6)
        fig.colorbar(im, ax=ax, label="Korelasi")
        ax.set_title("Heatmap korelasi 40 indikator")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

        return path


# ==========================================================
# BLOK PENGECEKAN MANDIRI
# ==========================================================
if __name__ == "__main__":
    from optimizer import DatasetOptimizer

    optimizer = DatasetOptimizer()
    df_final, loadings_final, report_final, n_attempts = optimizer.run(verbose=False)

    reporter = ReportGenerator(df_final, report_final)
    text_path = reporter.write_text_report()
    heatmap_path = reporter.save_heatmap()

    print("=== CEK REPORT.PY ===")
    print(f"Laporan teks tersimpan di : {text_path}")
    print(f"Heatmap tersimpan di      : {heatmap_path}")
