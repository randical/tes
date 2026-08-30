"""
optimizer.py
------------
Mengulang seluruh siklus pembangkitan sampai dataset lolos DUA LAPIS
pemeriksaan, bukan satu.

    Lapis satu, quality_control.py    cepat, pendekatan komponen utama
    Lapis dua,  validate_cfa.py       CFA sungguhan lewat semopy

Di proyek sebelumnya lapis dua dijalankan manual setelah main.py.
Pengukuran di sandbox menunjukkan satu siklus penuh termasuk CFA hanya
memakan sekitar 0,15 detik pada model 27 indikator dengan n=400,
sehingga lapis dua muat di dalam loop dan tidak perlu dijalankan
terpisah lagi.

BLOK SEED
Tiap percobaan mendapat blok seed sendiri, base = RANDOM_SEED +
percobaan x 100, lalu ditambah nol sampai lima untuk tiap modul.
Tanpa ini, dua percobaan bisa membangkitkan data yang identik dan loop
berputar sia-sia.
"""

import warnings

warnings.filterwarnings("ignore")

import config
from latent import LatentGenerator
from structural import StructuralModel
from indicator import IndicatorGenerator
from ordinal_engine import OrdinalEngine
from human_response import HumanResponseInjector
from demographics import DemographicGenerator
from quality_control import QualityChecker


class DatasetOptimizer:

    def __init__(self):
        self.riwayat = []

    @staticmethod
    def _one_attempt(base_seed):
        df_exo = LatentGenerator(seed=base_seed).sample()

        # Demografi dibangkitkan SEBELUM tahap struktural, karena kolomnya
        # bisa dipakai sebagai kovariat maupun sebagai variabel pengelompok
        # untuk moderasi kategorik.
        df_demo = df_kode = None
        if config.GENERATE_DEMOGRAPHICS:
            demo = DemographicGenerator(seed=base_seed + 5)
            df_demo = demo.generate(df_latent=df_exo)
            df_kode = demo.codes      # angka 1..k, inilah yang dipakai mesin

        model = StructuralModel(seed=base_seed + 1)
        df_latent = model.apply(df_exo, df_kode)

        gen = IndicatorGenerator(df_latent, seed=base_seed + 2)
        df_cont, loadings = gen.generate_all()

        inj = HumanResponseInjector(seed=base_seed + 4)
        df_cont = inj.apply_continuous(df_cont)
        df_likert = OrdinalEngine(seed=base_seed + 3).convert_dataframe(df_cont)
        df_likert = inj.apply_ordinal(df_likert)

        return df_likert, df_latent, loadings, model, gen, inj, df_demo, df_kode

    @staticmethod
    def _check_fit(df_likert):
        import validate_cfa
        hasil = validate_cfa.report(df_likert, verbose=False)
        fit = hasil["fit"]
        cmindf = fit["chi2"] / fit["DoF"] if fit["DoF"] else 99.0
        t = config.FIT_THRESHOLDS
        lolos = (cmindf <= t["CMIN/DF_max"]
                 and fit["CFI"] >= t["CFI_min"]
                 and fit["TLI"] >= t["TLI_min"]
                 and fit["RMSEA"] <= t["RMSEA_max"])
        ringkas = {"CMIN/DF": round(cmindf, 3), "CFI": fit["CFI"],
                   "TLI": fit["TLI"], "RMSEA": fit["RMSEA"],
                   "GFI": fit["GFI"], "AGFI": fit["AGFI"]}
        # Sekalian periksa validitas diskriminan pada korelasi LATEN
        disc = all(v["passed"] for v in hasil["discriminant"].values())
        return lolos and disc, ringkas, hasil

    def run(self, verbose=True):
        for percobaan in range(1, config.MAX_OPTIMIZER_ITERATIONS + 1):
            base = config.RANDOM_SEED + percobaan * 100
            paket = self._one_attempt(base)
            df_likert, df_latent, loadings, model, gen, inj, df_demo, df_kode = paket

            lap1 = QualityChecker(df_likert, loadings).run_all()
            catatan = {"percobaan": percobaan, "seed": base,
                       "lapis1": lap1["overall_passed"]}

            if not lap1["overall_passed"]:
                gagal = [k for k, v in lap1.items()
                         if isinstance(v, dict) and not v["passed"]]
                catatan["gagal_lapis1"] = gagal
                self.riwayat.append(catatan)
                if verbose:
                    print(f"  percobaan {percobaan:>3}  lapis 1 GAGAL pada {gagal}")
                continue

            if not config.FIT_CHECK_IN_OPTIMIZER:
                if verbose:
                    print(f"  percobaan {percobaan:>3}  lapis 1 LOLOS "
                          "(lapis 2 dilewati sesuai pengaturan)")
                self.riwayat.append(catatan)
                return dict(df_likert=df_likert, df_latent=df_latent,
                            loadings=loadings, model=model, generator=gen,
                            injector=inj, qc=lap1, cfa=None, demografi=df_demo, kode_demografi=df_kode,
                            seed=base, attempts=percobaan)

            fit_ok, ringkas, cfa = self._check_fit(df_likert)
            catatan.update({"lapis2": fit_ok, "fit": ringkas})
            self.riwayat.append(catatan)

            if verbose:
                status = "LOLOS" if fit_ok else "gagal"
                print(f"  percobaan {percobaan:>3}  lapis 1 LOLOS  "
                      f"lapis 2 {status}  {ringkas}")

            if fit_ok:
                return dict(df_likert=df_likert, df_latent=df_latent,
                            loadings=loadings, model=model, generator=gen,
                            injector=inj, qc=lap1, cfa=cfa, demografi=df_demo, kode_demografi=df_kode,
                            seed=base, attempts=percobaan)

        raise RuntimeError(
            f"Tidak ada dataset yang lolos dua lapis setelah "
            f"{config.MAX_OPTIMIZER_ITERATIONS} percobaan. Ini bukti STRUKTURAL, "
            "bukan kesialan. Periksa Modul 4 bagian tabel mitigasi sebelum "
            "sekadar menaikkan MAX_OPTIMIZER_ITERATIONS.")


if __name__ == "__main__":
    print("=== CEK OPTIMIZER.PY ===")
    hasil = DatasetOptimizer().run()
    print(f"\nBerhasil pada percobaan ke-{hasil['attempts']}, seed {hasil['seed']}")
    print("Ukuran dataset:", hasil["df_likert"].shape)
