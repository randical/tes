"""
quality_control.py
------------------
LAPIS SATU. Saringan cepat, dipanggil ratusan kali oleh optimizer.py.
Bukan pengganti CFA sungguhan. Lapis dua ada di validate_cfa.py.

DUA PERBAIKAN BUG HISTORIS YANG WAJIB DIPERTAHANKAN

Bug pertama. Versi lama menghitung AVE dan CR dari loading RANCANGAN
di config.py. Loading rancangan adalah parameter populasi yang belum
kena error pengukuran dan belum kena atenuasi konversi Likert. Modul
ini pernah melaporkan AVE 0,614 LOLOS sementara AMOS memberi 0,466
GAGAL pada dataset yang sama. Perbaikannya, loading diestimasi
LANGSUNG DARI DATA.

Bug kedua. Versi lama memeriksa validitas diskriminan memakai korelasi
antar SKOR KOMPOSIT. Skor komposit masih mengandung error pengukuran
yang meredam korelasinya. Modul ini pernah melaporkan LOLOS pada
korelasi komposit 0,671 sementara korelasi LATEN di AMOS ternyata
0,774 dan GAGAL. Perbaikannya, korelasi komposit dikoreksi dengan
rumus disattenuation ditambah margin aman.
"""

import numpy as np
import pandas as pd

import config


class QualityChecker:

    def __init__(self, df_likert, design_loadings=None):
        self.df = df_likert
        self.design_loadings = design_loadings or {}

    # ------------------------------------------------------
    # ESTIMASI LOADING DARI DATA (perbaikan bug pertama)
    # ------------------------------------------------------
    @staticmethod
    def _estimate_loadings_from_data(df, items):
        """
        Ambil komponen utama pertama dari matriks korelasi antar item
        satu konstruk. Untuk konstruk unidimensional, hasilnya sangat
        dekat dengan loading CFA.

            loading_item = eigenvector_item x akar(eigenvalue_terbesar)
        """
        korelasi = df[items].corr().to_numpy()
        eigval, eigvec = np.linalg.eigh(korelasi)
        idx = int(np.argmax(eigval))
        loading = eigvec[:, idx] * np.sqrt(eigval[idx])
        if loading.mean() < 0:      # arah eigenvector ambigu
            loading = -loading
        return np.clip(loading, -0.999, 0.999)

    @staticmethod
    def _cr_ave(loading):
        sum_l = loading.sum()
        sum_l2 = (loading ** 2).sum()
        sum_err = (1 - loading ** 2).sum()
        cr = (sum_l ** 2) / ((sum_l ** 2) + sum_err)
        ave = sum_l2 / len(loading)
        return float(cr), float(ave)

    # ------------------------------------------------------
    # PEMERIKSAAN
    # ------------------------------------------------------
    def check_duplicates(self):
        n = int(self.df.duplicated().sum())
        return {"n_duplicate": n, "passed": n <= config.MAX_DUPLICATE_ROWS}

    def check_distribution(self):
        nilai = self.df.to_numpy(dtype=float)
        mean = float(np.nanmean(nilai))
        sd = float(np.nanstd(nilai))
        lolos = (config.TARGET_MEAN[0] <= mean <= config.TARGET_MEAN[1]
                 and config.TARGET_STD[0] <= sd <= config.TARGET_STD[1])
        return {"mean": round(mean, 3), "sd": round(sd, 3), "passed": lolos}

    def check_reliability(self):
        hasil, lolos = {}, True
        for konstruk, items in config.CONSTRUCTS.items():
            sub = self.df[items].astype(float)
            k = len(items)
            alpha = (k / (k - 1)) * (1 - sub.var(ddof=1).sum() / sub.sum(axis=1).var(ddof=1))
            alpha = float(alpha)
            ok = config.CRONBACH_MIN <= alpha <= config.CRONBACH_MAX
            hasil[konstruk] = {"alpha": round(alpha, 3), "passed": ok}
            lolos = lolos and ok
        return {"per_construct": hasil, "passed": lolos}

    def check_validity(self):
        hasil, lolos = {}, True
        for konstruk, items in config.CONSTRUCTS.items():
            est = self._estimate_loadings_from_data(self.df, items)
            cr, ave = self._cr_ave(est)
            # Potong margin optimisme sebelum dibandingkan ke ambang.
            # Estimasi komponen utama bias ke ATAS terhadap CFA
            # sungguhan, jadi lapis satu tanpa potongan ini akan
            # meloloskan konstruk yang gagal di lapis dua.
            margin = getattr(config, "PCA_OPTIMISM_MARGIN", 0.0)
            ave_konservatif = ave - margin
            ok = cr >= config.CR_MIN and ave_konservatif >= config.AVE_MIN
            baris = {
                "CR": round(cr, 3),
                "AVE": round(ave, 3),
                "AVE_konservatif": round(ave_konservatif, 3),
                "loading_estimasi_min": round(float(est.min()), 3),
                "loading_estimasi_mean": round(float(est.mean()), 3),
                "passed": ok,
            }
            if konstruk in self.design_loadings:
                rancangan = np.array(list(self.design_loadings[konstruk].values()))
                baris["loading_rancangan_mean"] = round(float(rancangan.mean()), 3)
                baris["selisih_vs_rancangan"] = round(
                    float(est.mean() - rancangan.mean()), 3)
            hasil[konstruk] = baris
            lolos = lolos and ok
        return {"per_construct": hasil, "passed": lolos}

    def check_discriminant_validity(self, validity=None):
        """
        Perbaikan bug kedua. Korelasi antar skor komposit dikoreksi
        menjadi perkiraan korelasi laten lewat rumus disattenuation.

            r_laten = r_komposit / akar(CR_A x CR_B)

        Margin aman 0,05 ditambahkan karena rumus ini sendiri masih
        sedikit meremehkan angka CFA sungguhan. Pada kasus nyata
        selisihnya pernah mencapai 0,036.
        """
        validity = validity or self.check_validity()
        per = validity["per_construct"]

        komposit = pd.DataFrame({
            k: self.df[items].astype(float).mean(axis=1)
            for k, items in config.CONSTRUCTS.items()
        })
        r_komposit = komposit.corr()

        hasil, lolos, pelanggaran = {}, True, []
        for a in config.CONSTRUCTS:
            akar_ave = float(np.sqrt(per[a]["AVE_konservatif"]))
            tertinggi, lawan = 0.0, None
            for b in config.CONSTRUCTS:
                if a == b:
                    continue
                r_lat = abs(float(r_komposit.loc[a, b])) / np.sqrt(
                    max(per[a]["CR"] * per[b]["CR"], 1e-9))
                r_lat = min(r_lat, 0.999)
                if r_lat > tertinggi:
                    tertinggi, lawan = r_lat, b
            ok = akar_ave > (tertinggi + config.DISCRIMINANT_SAFETY_MARGIN)
            hasil[a] = {
                "akar_AVE": round(akar_ave, 3),
                "korelasi_laten_tertinggi": round(tertinggi, 3),
                "lawan": lawan,
                "passed": ok,
            }
            if not ok:
                pelanggaran.append(f"{a}-{lawan}")
            lolos = lolos and ok
        return {"per_construct": hasil, "violations": pelanggaran, "passed": lolos}

    def run_all(self):
        validity = self.check_validity()
        laporan = {
            "duplicates": self.check_duplicates(),
            "distribution": self.check_distribution(),
            "reliability": self.check_reliability(),
            "validity": validity,
            "discriminant": self.check_discriminant_validity(validity),
        }
        laporan["overall_passed"] = all(v["passed"] for v in laporan.values())
        return laporan


if __name__ == "__main__":
    from latent import LatentGenerator
    from structural import StructuralModel
    from indicator import IndicatorGenerator
    from ordinal_engine import OrdinalEngine
    from human_response import HumanResponseInjector

    df_full = StructuralModel().apply(LatentGenerator().sample())
    df_cont, loadings = IndicatorGenerator(df_full).generate_all()
    inj = HumanResponseInjector()
    df_likert = inj.apply_ordinal(OrdinalEngine().convert_dataframe(
        inj.apply_continuous(df_cont)))

    lap = QualityChecker(df_likert, loadings).run_all()

    print("=== CEK QUALITY_CONTROL.PY ===")
    print("duplikat   :", lap["duplicates"])
    print("distribusi :", lap["distribution"])
    print("\nValiditas konvergen, rancangan versus estimasi dari data")
    print(f"{'Konstruk':<6}{'rancangan':>11}{'PCA':>9}{'selisih':>9}"
          f"{'CR':>8}{'AVE':>8}{'AVE-margin':>12}{'status':>9}")
    for k, v in lap["validity"]["per_construct"].items():
        print(f"{k:<6}{v.get('loading_rancangan_mean', 0):>11.3f}"
              f"{v['loading_estimasi_mean']:>9.3f}"
              f"{v.get('selisih_vs_rancangan', 0):>+9.3f}"
              f"{v['CR']:>8.3f}{v['AVE']:>8.3f}{v['AVE_konservatif']:>12.3f}"
              f"{'OK' if v['passed'] else 'GAGAL':>9}")

    print("\nReliabilitas:", {k: v["alpha"]
                              for k, v in lap["reliability"]["per_construct"].items()})
    print("\nValiditas diskriminan")
    for k, v in lap["discriminant"]["per_construct"].items():
        print(f"  {k:<5} akar AVE {v['akar_AVE']:.3f} vs korelasi laten "
              f"{v['korelasi_laten_tertinggi']:.3f} ({v['lawan']}) "
              f"-> {'OK' if v['passed'] else 'GAGAL'}")

    print("\nSTATUS KESELURUHAN:", "LOLOS" if lap["overall_passed"] else "GAGAL")
