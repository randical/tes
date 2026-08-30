"""
quality_control.py
-------------------
Petugas QA: memeriksa apakah dataset yang dihasilkan sudah
memenuhi target di config.py (duplikasi, distribusi, reliabilitas,
validitas). Modul ini TIDAK mengubah data, hanya melaporkan
lolos/gagal -- keputusan untuk mengulang generate ada di optimizer.py.

CATATAN PERBAIKAN:
AVE dan CR sekarang dihitung dari LOADING YANG DIESTIMASI DARI DATA
LIKERT sungguhan (lihat _estimate_loadings_from_data), bukan dari
loading "rancangan" yang dipakai indicator.py untuk membuat data.
Ini penting karena proses pembulatan ke skala Likert 1-5
(ordinal_engine.py) dan error pengukuran SELALU melemahkan loading
asli -- kalau QC hanya mengecek loading rancangan, dia bisa bilang
"LOLOS" padahal analisis sungguhan di AMOS/SPSS nanti bilang "GAGAL".
"""

import numpy as np
import pandas as pd

import config


class QualityChecker:
    """
    Membungkus 4 jenis pemeriksaan (duplikasi, distribusi,
    reliabilitas, validitas) dan menggabungkannya jadi satu
    laporan lengkap lewat run_all().
    """

    def __init__(self, df_likert, loadings):
        # df_likert: hasil akhir dari ordinal_engine.py (tabel Likert)
        # loadings : dictionary loading RANCANGAN dari indicator.py --
        #            sekarang hanya dipakai sebagai pembanding di
        #            laporan, bukan untuk menghitung AVE/CR lagi.
        self.df = df_likert
        self.loadings = loadings

    def check_duplicates(self):
        """
        KENAPA:
        Dataset survei sungguhan tidak boleh punya dua responden
        dengan jawaban 40 kolom yang identik persis -- itu tanda
        data cacat/duplikat, bukan variasi manusia yang wajar.

        BAGAIMANA:
        df.duplicated() menandai baris mana yang sama persis dengan
        baris sebelumnya; .sum() menghitung berapa banyak.
        """
        n_duplicate = self.df.duplicated().sum()
        passed = n_duplicate <= config.MAX_DUPLICATE_ROWS
        return {"n_duplicate": int(n_duplicate), "passed": passed}

    def check_distribution(self):
        """
        KENAPA:
        Kita menargetkan rata-rata dan simpangan baku tertentu di
        config.py (TARGET_MEAN, TARGET_STD) supaya data terasa seperti
        survei sungguhan yang condong ke jawaban positif, bukan acak
        merata.

        BAGAIMANA:
        Hitung rata-rata dan simpangan baku dari SELURUH sel data
        (semua 200x40 angka sekaligus), lalu cek apakah masuk rentang
        target.
        """
        mean = self.df.values.mean()
        std = self.df.values.std()
        mean_passed = config.TARGET_MEAN[0] <= mean <= config.TARGET_MEAN[1]
        std_passed = config.TARGET_STD[0] <= std <= config.TARGET_STD[1]
        return {
            "mean": mean, "std": std,
            "mean_passed": mean_passed, "std_passed": std_passed,
        }

    def _cronbach_alpha(self, items_df):
        """
        Rumus Cronbach's Alpha standar -- sudah familiar untuk Anda,
        di sini hanya diterjemahkan ke sintaks pandas/numpy.
        """
        n_items = items_df.shape[1]
        var_total = items_df.sum(axis=1).var(ddof=1)
        var_sum_items = items_df.var(axis=0, ddof=1).sum()
        alpha = (n_items / (n_items - 1)) * (1 - var_sum_items / var_total)
        return alpha

    def check_reliability(self):
        """
        KENAPA:
        Tiap konstruk (PE, EE, dst) harus punya Alpha dalam rentang
        target -- terlalu rendah berarti item-itemnya tidak konsisten
        mengukur hal yang sama, terlalu tinggi (>0.95 misalnya) malah
        mencurigakan (item-itemnya nyaris identik/redundan).

        BAGAIMANA:
        Hitung Alpha untuk tiap konstruk satu per satu, simpan hasil
        dan status lolos/gagalnya dalam satu dictionary per konstruk.
        """
        per_construct = {}
        for construct, items in config.CONSTRUCTS.items():
            alpha = self._cronbach_alpha(self.df[items])
            passed = config.CRONBACH_MIN <= alpha <= config.CRONBACH_MAX
            per_construct[construct] = {"alpha": alpha, "passed": passed}

        overall_passed = all(res["passed"] for res in per_construct.values())
        return {"per_construct": per_construct, "overall_passed": overall_passed}

    def _calculate_cr_ave(self, item_loadings):
        """
        Rumus Composite Reliability dan AVE standar, sama seperti
        yang keluar dari output SmartPLS/AMOS.
        """
        loadings_arr = np.array(item_loadings)
        sum_loading = loadings_arr.sum()
        sum_error = np.sum(1 - loadings_arr ** 2)
        cr = (sum_loading ** 2) / (sum_loading ** 2 + sum_error)
        ave = np.mean(loadings_arr ** 2)
        return cr, ave

    def _estimate_loadings_from_data(self, items):
        """
        KENAPA:  # <-- BARU
        Loading yang dipakai untuk MEMBUAT data (di indicator.py) itu
        loading "rancangan" -- bukan loading yang akan ditemukan CFA
        sungguhan (AMOS/SPSS) kalau data itu dianalisis. Antara
        rancangan dan hasil analisis ada dua sumber pelemahan
        (atenuasi):
          1. Error pengukuran yang sengaja ditambahkan (supaya data
             terasa manusiawi, bukan sempurna).
          2. Pembulatan skor kontinu jadi kategori Likert 1-5
             (ordinal_engine.py) -- proses ini SELALU melemahkan
             korelasi asli, walau sedikit.
        Kalau AVE/CR dihitung dari loading rancangan, hasilnya bisa
        "LOLOS" padahal AMOS nanti bilang "GAGAL" (ini yang terjadi
        pada konstruk FC kemarin). Supaya QC ini bisa dipercaya, dia
        harus menghitung loading dari DATA yang benar-benar jadi --
        sama seperti cara kerja AMOS.

        BAGAIMANA:
        Untuk 5 kolom satu konstruk, hitung matriks korelasi antar
        item, lalu ambil komponen utama pertamanya (principal
        component) lewat dekomposisi eigen. Loading tiap item
        didekati dengan:
            loading_item = eigenvector_item * akar(eigenvalue_terbesar)
        Ini teknik factor analysis satu-faktor standar (principal
        axis factoring) -- sangat dekat dengan hasil CFA untuk
        konstruk unidimensional seperti punya kita.
        """
        item_corr = self.df[items].corr().values
        eigenvalues, eigenvectors = np.linalg.eigh(item_corr)

        # np.linalg.eigh mengurutkan dari KECIL ke BESAR -- ambil yang
        # terakhir (terbesar), itu komponen utama pertama.
        largest_idx = np.argmax(eigenvalues)
        largest_eigenvalue = eigenvalues[largest_idx]
        largest_eigenvector = eigenvectors[:, largest_idx]

        estimated_loadings = largest_eigenvector * np.sqrt(largest_eigenvalue)

        # Arah eigenvector itu ambigu (bisa "kebalik" semua tandanya) --
        # loading yang benar harus positif, karena semua item dirancang
        # searah dengan konstruknya.
        if estimated_loadings.mean() < 0:
            estimated_loadings = -estimated_loadings

        return estimated_loadings

    def check_validity(self):
        """
        KENAPA:
        CR memastikan seluruh item dalam konstruk secara bersama-sama
        cukup reliabel; AVE memastikan konstruk menjelaskan lebih
        banyak variansi item dibanding error-nya (syarat validitas
        konvergen standar dalam SEM).

        BAGAIMANA:  # <-- DIUBAH
        Untuk tiap konstruk, loading dihitung LANGSUNG DARI DATA
        Likert lewat _estimate_loadings_from_data() -- bukan dari
        self.loadings rancangan lagi. Loading rancangan tetap
        disimpan sebagai "design_loading_mean" di laporan, supaya
        Anda bisa lihat sendiri seberapa jauh atenuasinya dibanding
        "estimated_loading_mean" (hasil estimasi dari data).
        """
        per_construct = {}
        for construct, items in config.CONSTRUCTS.items():
            estimated_loadings = self._estimate_loadings_from_data(items)
            cr, ave = self._calculate_cr_ave(estimated_loadings)

            design_loadings = [self.loadings[construct][item] for item in items]

            cr_passed = cr >= config.CR_MIN
            ave_passed = ave >= config.AVE_MIN
            per_construct[construct] = {
                "cr": cr, "ave": ave,
                "cr_passed": cr_passed, "ave_passed": ave_passed,
                "design_loading_mean": float(np.mean(design_loadings)),
                "estimated_loading_mean": float(np.mean(estimated_loadings)),
            }

        overall_passed = all(
            res["cr_passed"] and res["ave_passed"] for res in per_construct.values()
        )
        return {"per_construct": per_construct, "overall_passed": overall_passed}

    def check_discriminant_validity(self):
        """
        KENAPA:
        AVE dan CR tinggi saja belum cukup -- syarat validitas
        diskriminan (Fornell-Larcker) memastikan tiap konstruk lebih
        mirip DIRINYA SENDIRI (lewat item-itemnya) dibanding konstruk
        lain. Kalau tidak, dua konstruk yang seharusnya beda malah
        tumpang tindih -- item-itemnya sebenarnya mengukur hal yang
        sama.

        KENAPA PAKAI KOREKSI DISATTENUATION:  # <-- BARU
        Korelasi antar SKOR KOMPOSIT (rata-rata item mentah) SELALU
        lebih lemah dibanding korelasi antar KONSTRUK LATEN sungguhan
        -- composite score masih mengandung error pengukuran yang
        "meredam" korelasinya, persis seperti loading yang melemah
        kalau tidak dikoreksi (lihat _estimate_loadings_from_data).
        Kalau dibiarkan tanpa koreksi, QC ini bisa bilang "LOLOS"
        padahal CFA sungguhan (AMOS) bilang "GAGAL" -- ini PERSIS yang
        terjadi pada pasangan EC-FC di data Anda (korelasi composite
        0.671, tapi korelasi CFA sungguhan 0.774 -- cukup untuk
        membuat keduanya gagal discriminant validity).
        Rumus koreksinya (disattenuation / koreksi Spearman):
            r_laten ~ r_composite / sqrt(CR_A x CR_B)
        Tapi rumus ini masih sedikit meremehkan angka CFA sungguhan
        (pada pengujian nyata, selisihnya sampai ~0.04) -- makanya
        ditambah MARGIN AMAN (config.DISCRIMINANT_SAFETY_MARGIN)
        supaya kasus yang pas di garis batas tetap tertangkap sebagai
        "GAGAL" di sini, bukan baru ketahuan setelah buka AMOS.

        BAGAIMANA:
        Untuk tiap konstruk, hitung korelasi composite score dengan
        semua konstruk lain, koreksi tiap angkanya dengan rumus
        disattenuation, ambil yang PALING TINGGI, lalu bandingkan
        (sqrt(AVE) harus lebih besar dari korelasi terkoreksi + margin).
        """
        composite = pd.DataFrame({
            c: self.df[items].mean(axis=1) for c, items in config.CONSTRUCTS.items()
        })
        corr = composite.corr()

        validity_report = self.check_validity()

        per_construct = {}
        for construct in config.CONSTRUCTS:
            ave = validity_report["per_construct"][construct]["ave"]
            cr_self = validity_report["per_construct"][construct]["cr"]
            sqrt_ave = ave ** 0.5

            max_disattenuated_corr = 0.0
            max_partner = None
            for other in config.CONSTRUCTS:
                if other == construct:
                    continue
                observed_corr = corr.loc[construct, other]
                cr_other = validity_report["per_construct"][other]["cr"]
                # Koreksi disattenuation -- dibatasi maksimum 0.999 supaya
                # tidak pernah "lewat" dari batas korelasi maksimum (1.0)
                # akibat pembagian dengan angka CR yang kecil.
                disattenuated = min(observed_corr / np.sqrt(cr_self * cr_other), 0.999)
                if disattenuated > max_disattenuated_corr:
                    max_disattenuated_corr = disattenuated
                    max_partner = other

            passed = sqrt_ave > (max_disattenuated_corr + config.DISCRIMINANT_SAFETY_MARGIN)
            per_construct[construct] = {
                "sqrt_ave": sqrt_ave,
                "max_corr": max_disattenuated_corr,
                "max_corr_partner": max_partner,
                "passed": passed,
            }

        overall_passed = all(res["passed"] for res in per_construct.values())
        return {"per_construct": per_construct, "overall_passed": overall_passed}

    def run_all(self):
        """
        KENAPA:
        Ini satu-satunya method yang nanti dipanggil oleh optimizer.py.
        Optimizer tidak perlu tahu detail 4 pemeriksaan di atas --
        dia cukup bertanya satu hal: report["overall_passed"], True
        atau False.

        BAGAIMANA:
        Jalankan keempat pemeriksaan, lalu gabungkan semua status
        lolos/gagalnya jadi satu kesimpulan akhir dengan all().
        """
        duplicate = self.check_duplicates()
        distribution = self.check_distribution()
        reliability = self.check_reliability()
        validity = self.check_validity()
        discriminant = self.check_discriminant_validity()

        overall_passed = all([
            duplicate["passed"],
            distribution["mean_passed"],
            distribution["std_passed"],
            reliability["overall_passed"],
            validity["overall_passed"],
            discriminant["overall_passed"],
        ])

        return {
            "overall_passed": overall_passed,
            "duplicate": duplicate,
            "distribution": distribution,
            "reliability": reliability,
            "validity": validity,
            "discriminant": discriminant,
        }


# ==========================================================
# BLOK PENGECEKAN MANDIRI
# ==========================================================
if __name__ == "__main__":
    from latent import LatentGenerator
    from structural import StructuralModel
    from indicator import IndicatorGenerator
    from ordinal_engine import OrdinalEngine

    latent_gen = LatentGenerator()
    df_exogenous = latent_gen.sample_latent()
    df_latent = StructuralModel().apply(df_exogenous)

    indicator_gen = IndicatorGenerator(df_latent)
    df_continuous, loadings = indicator_gen.generate_all()

    ordinal = OrdinalEngine()
    df_likert = ordinal.convert_dataframe(df_continuous)

    checker = QualityChecker(df_likert, loadings)
    report = checker.run_all()

    print("=== CEK QUALITY_CONTROL.PY ===")
    status = "LOLOS" if report["overall_passed"] else "GAGAL"
    print(f"Hasil keseluruhan: {status}\n")

    print(f"Duplikat baris : {report['duplicate']['n_duplicate']} "
          f"(syarat: maks {config.MAX_DUPLICATE_ROWS})")

    d = report["distribution"]
    mean_status = "OK" if d["mean_passed"] else "MELESET"
    std_status = "OK" if d["std_passed"] else "MELESET"
    print(f"Rata-rata      : {d['mean']:.3f} (target {config.TARGET_MEAN}) -> {mean_status}")
    print(f"Simpangan baku : {d['std']:.3f} (target {config.TARGET_STD}) -> {std_status}")

    print("\nCronbach's Alpha per konstruk:")
    for construct, res in report["reliability"]["per_construct"].items():
        status_c = "OK" if res["passed"] else "MELESET"
        print(f"  {construct}: {res['alpha']:.3f} -> {status_c}")

    print("\nComposite Reliability & AVE per konstruk (dari data, bukan rancangan):")  # <-- DIUBAH
    for construct, res in report["validity"]["per_construct"].items():
        status_c = "OK" if (res["cr_passed"] and res["ave_passed"]) else "MELESET"
        gap = res["design_loading_mean"] - res["estimated_loading_mean"]  # <-- BARU
        print(f"  {construct}: CR={res['cr']:.3f}, AVE={res['ave']:.3f} -> {status_c}  "
              f"(loading rancangan={res['design_loading_mean']:.3f}, "
              f"loading data={res['estimated_loading_mean']:.3f}, atenuasi={gap:.3f})")  # <-- BARU

    print("\nDiscriminant Validity (Fornell-Larcker, sudah dikoreksi disattenuation):")
    for construct, res in report["discriminant"]["per_construct"].items():
        status_c = "OK" if res["passed"] else "MELESET"
        print(f"  {construct}: sqrt(AVE)={res['sqrt_ave']:.3f} vs korelasi tertinggi="
              f"{res['max_corr']:.3f} (dgn {res['max_corr_partner']}) -> {status_c}")
