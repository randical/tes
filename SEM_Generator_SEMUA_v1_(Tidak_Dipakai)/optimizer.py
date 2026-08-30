"""
optimizer.py
------------
Mengulang siklus latent.py -> indicator.py -> ordinal_engine.py ->
quality_control.py dengan seed berbeda tiap percobaan, sampai semua
syarat di config.py lolos, atau sampai batas maksimum percobaan
tercapai (supaya tidak mengulang selamanya kalau targetnya memang
tidak mungkin tercapai).
"""

from latent import LatentGenerator
from structural import StructuralModel
from indicator import IndicatorGenerator
from ordinal_engine import OrdinalEngine
from quality_control import QualityChecker

import config


class DatasetOptimizer:
    """
    Membungkus satu siklus penuh generate + cek kualitas, dan
    mengulanginya lewat while loop sampai lolos atau kehabisan jatah
    percobaan.
    """

    def __init__(self):
        self.max_attempts = config.MAX_OPTIMIZER_ITERATIONS

    def _run_one_attempt(self, attempt_number):
        """
        KENAPA:
        Setiap percobaan harus menghasilkan data yang BENAR-BENAR baru,
        bukan pengulangan angka yang sama. Karena itu tiap percobaan
        dapat "jatah seed" sendiri yang tidak akan pernah tabrakan
        dengan percobaan lain.

        BAGAIMANA:
        base_seed dihitung dari nomor percobaan (dikali 100 supaya ada
        banyak ruang kosong di antaranya). Tiap tahap (latent,
        structural, indicator, ordinal) tetap dapat seed yang sedikit
        berbeda satu sama lain.
        """
        base_seed = config.RANDOM_SEED + (attempt_number * 100)

        latent_gen = LatentGenerator(seed=base_seed)
        df_exogenous = latent_gen.sample_latent()

        structural = StructuralModel(seed=base_seed + 1)
        df_latent = structural.apply(df_exogenous)

        indicator_gen = IndicatorGenerator(df_latent, seed=base_seed + 2)
        df_continuous, loadings = indicator_gen.generate_all()

        ordinal = OrdinalEngine(seed=base_seed + 3)
        df_likert = ordinal.convert_dataframe(df_continuous)

        checker = QualityChecker(df_likert, loadings)
        report = checker.run_all()

        return df_likert, loadings, report

    def run(self, verbose=True):
        """
        KENAPA:
        Ini inti optimizer: coba generate, cek lolos atau tidak, kalau
        belum ulangi dengan seed baru -- sampai lolos atau jatah
        percobaan habis.

        BAGAIMANA:
        while loop dengan penghitung attempt yang bertambah tiap putaran.
        Begitu report["overall_passed"] bernilai True, langsung
        dikembalikan (loop berhenti otomatis karena ada return).
        """
        attempt = 1
        df_likert, loadings, report = None, None, None

        while attempt <= self.max_attempts:
            df_likert, loadings, report = self._run_one_attempt(attempt)

            if verbose:
                status = "LOLOS" if report["overall_passed"] else "belum lolos"
                print(f"Percobaan {attempt}: {status}")

            if report["overall_passed"]:
                if verbose:
                    print(f"\nBerhasil pada percobaan ke-{attempt}.")
                return df_likert, loadings, report, attempt

            attempt += 1

        if verbose:
            print(f"\nGagal mencapai target setelah {self.max_attempts} percobaan.")
        return df_likert, loadings, report, self.max_attempts


# ==========================================================
# BLOK PENGECEKAN MANDIRI
# ==========================================================
if __name__ == "__main__":
    optimizer = DatasetOptimizer()
    df_final, loadings_final, report_final, n_attempts = optimizer.run()

    print("\n=== CEK OPTIMIZER.PY ===")
    print(f"Jumlah percobaan : {n_attempts} dari maks {config.MAX_OPTIMIZER_ITERATIONS}")
    print(f"Status akhir     : {'LOLOS' if report_final['overall_passed'] else 'GAGAL'}")

    d = report_final["distribution"]
    print(f"Rata-rata akhir  : {d['mean']:.3f} (target {config.TARGET_MEAN})")
    print(f"SD akhir         : {d['std']:.3f} (target {config.TARGET_STD})")
