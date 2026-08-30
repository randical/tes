"""
structural.py
-------------
MESIN INTI ENGINE BARU.

Menghitung nilai tiap konstruk ENDOGEN dari prediktornya, mengikuti
deklarasi config.STRUCTURAL_MODEL, dalam urutan yang benar secara
otomatis.

Analogi web: penyortiran topologis di sini persis seperti cara bundler
JavaScript menentukan urutan mengeksekusi modul dari grafik import.
Anda menulis deklarasinya, urutannya dihitung mesin.

Tiga hal yang dikerjakan modul ini dan tidak ada di engine lama.

1. Penyortiran topologis, sehingga mediasi berjenjang otomatis benar.
2. Perhitungan variansi residual, sehingga tiap variabel endogen tetap
   berskala baku. Tanpa ini, variabel di ujung rantai mediasi punya
   variansi membengkak dan konversi Likert-nya terdistorsi.
3. Suku moderasi sebagai PRODUK KONTINU, bukan dua set koefisien
   terpisah per kelompok.
"""

import numpy as np
import pandas as pd

import config


class ModelSpecificationError(Exception):
    pass


class StructuralModel:

    def __init__(self, seed=None):
        seed = seed if seed is not None else config.RANDOM_SEED + 1
        self.rng = np.random.default_rng(seed)
        self.model = config.effective_structural_model()
        self.moderations = config.MODERATIONS
        self.group_moderations = getattr(config, "GROUP_MODERATIONS", [])
        self.covariates = list(getattr(config, "COVARIATES", []))
        self.diagnostics = {}
        self._residual_store = {}
        self._residual_pool = {}
        self._validate_specification()

    # ------------------------------------------------------
    # PEMERIKSAAN SPESIFIKASI
    # ------------------------------------------------------
    def _validate_specification(self):
        laten = set(config.latent_variables())
        sah = laten | set(self.covariates)

        # Konstruk orde dua
        for induk, dimensi in getattr(config, "SECOND_ORDER", {}).items():
            if induk in config.CONSTRUCTS:
                raise ModelSpecificationError(
                    f"Induk orde dua '{induk}' tidak boleh punya item sendiri di "
                    "CONSTRUCTS. Ia diukur lewat dimensinya.")
            for dim, load in dimensi.items():
                if dim not in config.CONSTRUCTS:
                    raise ModelSpecificationError(
                        f"Dimensi '{dim}' dari '{induk}' harus ada di CONSTRUCTS "
                        "lengkap dengan itemnya.")
                if dim in config.STRUCTURAL_MODEL:
                    raise ModelSpecificationError(
                        f"Dimensi '{dim}' tidak boleh jadi variabel endogen di "
                        "STRUCTURAL_MODEL. Nilainya sudah ditentukan induknya.")
                if not 0.0 < float(load) < 1.0:
                    raise ModelSpecificationError(
                        f"Loading '{induk}' ke '{dim}' harus di antara 0 dan 1.")

        # Kovariat teramati
        for c in self.covariates:
            if not getattr(config, "GENERATE_DEMOGRAPHICS", False):
                raise ModelSpecificationError(
                    f"Kovariat '{c}' diminta tapi GENERATE_DEMOGRAPHICS bernilai False.")
            if c not in config.DEMOGRAPHIC_SPEC:
                raise ModelSpecificationError(
                    f"Kovariat '{c}' tidak ada di DEMOGRAPHIC_SPEC.")
            if c in laten:
                raise ModelSpecificationError(
                    f"Nama '{c}' dipakai sekaligus sebagai konstruk laten dan kovariat.")

        for outcome, prediktor in self.model.items():
            if outcome not in laten:
                raise ModelSpecificationError(
                    f"'{outcome}' jadi variabel endogen tapi bukan variabel laten.")
            for p in prediktor:
                if p not in sah:
                    raise ModelSpecificationError(
                        f"Prediktor '{p}' untuk '{outcome}' bukan konstruk laten "
                        "maupun kovariat yang terdaftar di COVARIATES.")
                if p == outcome:
                    raise ModelSpecificationError(
                        f"'{outcome}' tidak boleh memprediksi dirinya sendiri.")

        for g in self.group_moderations:
            o, p, kol = g["outcome"], g["predictor"], g["group_column"]
            if o not in self.model:
                raise ModelSpecificationError(
                    f"Moderasi kategorik menunjuk outcome '{o}' yang bukan endogen.")
            if p not in self.model[o]:
                raise ModelSpecificationError(
                    f"Jalur '{p}' -> '{o}' harus punya efek utama sebelum dimoderasi.")
            if kol not in config.DEMOGRAPHIC_SPEC:
                raise ModelSpecificationError(
                    f"Kolom kelompok '{kol}' tidak ada di DEMOGRAPHIC_SPEC.")

        for m in self.moderations:
            o, p, w = m["outcome"], m["predictor"], m["moderator"]
            if o not in self.model:
                raise ModelSpecificationError(
                    f"Moderasi menunjuk outcome '{o}' yang bukan variabel endogen.")
            if p not in self.model[o]:
                raise ModelSpecificationError(
                    f"Jalur '{p}' -> '{o}' harus punya efek utama sebelum dimoderasi.")
            if w not in self.model[o]:
                raise ModelSpecificationError(
                    f"Moderator '{w}' wajib punya efek utama pada '{o}'. "
                    "Model interaksi tanpa efek utama moderator tidak dapat ditafsirkan.")

    # ------------------------------------------------------
    # PENYORTIRAN TOPOLOGIS
    # ------------------------------------------------------
    def topological_order(self):
        """
        KENAPA:
        Pada mediasi berjenjang X -> M1 -> M2 -> Y, nilai M2 tidak bisa
        dihitung sebelum M1 ada. Urutan penulisan di config.py tidak
        dijamin benar, jadi mesin harus menentukannya sendiri.

        BAGAIMANA:
        Algoritma Kahn. Ulangi terus: ambil variabel endogen yang semua
        prediktornya sudah siap, masukkan ke antrean, tandai siap.
        Kalau tersisa variabel yang tidak pernah siap, berarti ada
        lingkaran sebab-akibat dan model tidak sah.
        """
        siap = set(config.exogenous_constructs()) | set(self.covariates)
        sisa = dict(self.model)
        urutan = []

        while sisa:
            maju = False
            for outcome in list(sisa):
                if all(p in siap for p in sisa[outcome]):
                    urutan.append(outcome)
                    siap.add(outcome)
                    del sisa[outcome]
                    maju = True
            if not maju:
                raise ModelSpecificationError(
                    f"Ada lingkaran sebab-akibat di antara {list(sisa)}. "
                    "Model SEM rekursif tidak boleh punya siklus.")
        return urutan

    # ------------------------------------------------------
    # RESIDUAL BERKORELASI
    # ------------------------------------------------------
    def _draw_residual(self, outcome, n):
        """
        KENAPA:
        Pada mediasi paralel, dua mediator umumnya masih berbagi
        penyebab yang tidak masuk model. Di AMOS hal ini digambar
        sebagai panah dua arah antar error term. Kalau data yang
        dibuat generator TIDAK punya kovarians itu sementara model
        yang digambar punya, parameter tersebut akan diestimasi
        mendekati nol dan df model terbuang percuma.

        BAGAIMANA:
        Untuk tiap pasangan di RESIDUAL_COVARIANCES, residual kedua
        anggotanya dibangkitkan sekali sebagai pasangan berkorelasi,
        lalu diambil satu per satu saat gilirannya tiba.
        """
        if outcome in self._residual_pool:
            return self._residual_pool.pop(outcome)

        for spec in getattr(config, "RESIDUAL_COVARIANCES", []):
            a, b = spec["between"]
            if outcome not in (a, b):
                continue
            r = float(spec["correlation"])
            z1 = self.rng.normal(0, 1, n)
            z2 = r * z1 + np.sqrt(1 - r ** 2) * self.rng.normal(0, 1, n)
            pasangan = {a: z1, b: z2}
            self._residual_pool[b if outcome == a else a] = pasangan[b if outcome == a else a]
            return pasangan[outcome]

        return self.rng.normal(0, 1, n)

    # ------------------------------------------------------
    # PERHITUNGAN
    # ------------------------------------------------------
    def apply(self, df_exogenous, df_demografi=None):
        df = df_exogenous.copy()

        # Kovariat teramati dibakukan supaya koefisiennya terbaca sebagai
        # beta terstandardisasi, sama perlakuannya dengan konstruk laten.
        self.group_codes = {}
        if df_demografi is not None:
            for c in self.covariates:
                kol = df_demografi[c].to_numpy(dtype=float)
                df[c] = (kol - kol.mean()) / kol.std()
            for g in self.group_moderations:
                self.group_codes[g["group_column"]] = \
                    df_demografi[g["group_column"]].to_numpy()
        elif self.covariates or self.group_moderations:
            raise ModelSpecificationError(
                "COVARIATES atau GROUP_MODERATIONS dipakai, tapi apply() dipanggil "
                "tanpa df_demografi. Bangkitkan demografi lebih dulu.")

        for outcome in self.topological_order():
            koef = self.model[outcome]
            nama_prediktor = list(koef)
            beta = np.array([koef[p] for p in nama_prediktor], dtype=float)

            X = df[nama_prediktor].to_numpy(dtype=float)

            # Moderasi kategorik. Koefisien satu jalur dibuat berbeda per
            # kelompok, jadi beta menjadi vektor per responden, bukan skalar.
            beta_matriks = np.tile(beta, (len(df), 1))
            jalur_kelompok = []
            for g in self.group_moderations:
                if g["outcome"] != outcome:
                    continue
                idx = nama_prediktor.index(g["predictor"])
                kode = self.group_codes[g["group_column"]]
                tambahan = np.zeros(len(df))
                for k, d in g["deltas"].items():
                    tambahan[kode == k] = float(d)
                beta_matriks[:, idx] = beta_matriks[:, idx] + tambahan
                jalur_kelompok.append(
                    f'{g["predictor"]} x {g["group_column"]}')

            terjelaskan = (X * beta_matriks).sum(axis=1)

            suku_interaksi = []
            for m in self.moderations:
                if m["outcome"] != outcome:
                    continue
                p_kolom = df[m["predictor"]].to_numpy(dtype=float)
                w_kolom = df[m["moderator"]].to_numpy(dtype=float)

                # Pemusatan rerata ganda lalu penskalaan ke variansi 1,
                # supaya koefisien yang Anda tulis di config.py benar-benar
                # terbaca sebagai beta terstandardisasi.
                produk = (p_kolom - p_kolom.mean()) * (w_kolom - w_kolom.mean())
                produk = (produk - produk.mean()) / produk.std()

                terjelaskan = terjelaskan + m["coefficient"] * produk
                suku_interaksi.append(f'{m["predictor"]}x{m["moderator"]}')
            suku_interaksi.extend(jalur_kelompok)

            var_terjelaskan = float(np.var(terjelaskan))
            if var_terjelaskan >= 1.0:
                raise ModelSpecificationError(
                    f"Koefisien menuju '{outcome}' terlalu besar. "
                    f"Variansi terjelaskan {var_terjelaskan:.3f} sudah melampaui 1.0, "
                    "sehingga tidak tersisa ruang untuk error. Kecilkan koefisiennya.")

            sd_residual = np.sqrt(1.0 - var_terjelaskan)
            residual = self._draw_residual(outcome, len(df))
            # Dibakukan tepat, supaya R2 rancangan benar-benar tercapai
            # dan tidak tergerus derau pengambilan sampel.
            residual = (residual - residual.mean()) / residual.std() * sd_residual
            df[outcome] = terjelaskan + residual
            self._residual_store[outcome] = residual

            self.diagnostics[outcome] = {
                "prediktor": nama_prediktor,
                "beta_rancangan": dict(zip(nama_prediktor, beta.round(3))),
                "suku_interaksi": suku_interaksi,
                "R2_rancangan": round(var_terjelaskan, 4),
                "sd_residual": round(float(sd_residual), 4),
                "sd_hasil": round(float(df[outcome].std()), 4),
            }

        self.residual_corr = {}
        for spec in getattr(config, "RESIDUAL_COVARIANCES", []):
            a, b = spec["between"]
            if a in self._residual_store and b in self._residual_store:
                self.residual_corr[f"{a}~{b}"] = round(float(np.corrcoef(
                    self._residual_store[a], self._residual_store[b])[0, 1]), 4)

        return df

    def print_diagnostics(self):
        print("Urutan hitung topologis:", " -> ".join(self.topological_order()))
        for outcome, d in self.diagnostics.items():
            inter = (" + " + " + ".join(d["suku_interaksi"])) if d["suku_interaksi"] else ""
            print(f"\n  {outcome} <- {' + '.join(d['prediktor'])}{inter}")
            print(f"    beta rancangan : {d['beta_rancangan']}")
            print(f"    R2 rancangan   : {d['R2_rancangan']}")
            print(f"    sd residual    : {d['sd_residual']}   sd hasil: {d['sd_hasil']}")
        if getattr(self, "residual_corr", None):
            print("\n  Korelasi residual antar mediator:", self.residual_corr)


if __name__ == "__main__":
    from latent import LatentGenerator

    df_exo = LatentGenerator().sample()
    model = StructuralModel()
    df_full = model.apply(df_exo)

    print("=== CEK STRUCTURAL.PY ===")
    model.print_diagnostics()

    print("\nSimpangan baku semua konstruk (idealnya semua mendekati 1):")
    print(df_full.std().round(3).to_string())

    print("\nKorelasi antar konstruk:")
    print(df_full.corr().round(3).to_string())

    print("\n--- Uji lingkaran sebab-akibat (harus gagal terkendali) ---")
    simpan_m, simpan_mod, simpan_rc = (config.STRUCTURAL_MODEL,
                                       config.MODERATIONS,
                                       config.RESIDUAL_COVARIANCES)
    config.STRUCTURAL_MODEL = {"PN": {"AP": 0.4}, "AP": {"PN": 0.4}}
    config.MODERATIONS = []
    config.RESIDUAL_COVARIANCES = []
    try:
        StructuralModel().topological_order()
        print("  GAGAL: siklus tidak terdeteksi")
    except ModelSpecificationError as e:
        print("  Terdeteksi benar ->", e)

    print("\n--- Uji moderator tanpa efek utama (harus gagal terkendali) ---")
    config.STRUCTURAL_MODEL = {"PI": {"PN": 0.4}}
    config.MODERATIONS = [{"outcome": "PI", "predictor": "PN",
                           "moderator": "CIA", "coefficient": -0.2}]
    try:
        StructuralModel()
        print("  GAGAL: tidak terdeteksi")
    except ModelSpecificationError as e:
        print("  Terdeteksi benar ->", e)

    config.STRUCTURAL_MODEL, config.MODERATIONS, config.RESIDUAL_COVARIANCES = (
        simpan_m, simpan_mod, simpan_rc)
