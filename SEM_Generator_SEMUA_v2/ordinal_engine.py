"""
ordinal_engine.py
------------------
Mengubah skor kontinu menjadi kategori skala Likert lewat AMBANG BATAS
(threshold), bukan pembulatan.

KENAPA BUKAN PEMBULATAN
Pembulatan naif (misal round(z*0.8 + 4)) memampatkan ekor distribusi ke
kategori tepi dan memperlakukan jarak antar kategori sebagai sama besar
di ruang laten. Akibatnya korelasi antar item selalu turun lebih jauh
dari yang perlu. Pendekatan ambang batas menempatkan potongan justru di
titik yang menghasilkan proporsi kategori yang kita inginkan, sehingga
pelemahan korelasi ditekan seminimal mungkin.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

import config


# ==========================================================
# PENCARIAN PROPORSI NUMERIK
# ==========================================================
def props_from_normal(mu, sigma, k):
    """
    Diskretisasi distribusi normal menjadi k kategori.
    Titik potong berada di 1.5, 2.5, ... k-0.5.
    """
    cut = np.arange(1.5, k, 1.0)
    cdf = norm.cdf((cut - mu) / sigma)
    return np.diff(np.concatenate(([0.0], cdf, [1.0])))


def find_proportions(n_points, target_mean, target_sd, seed=0, verbose=False):
    """
    KENAPA:
    Menebak proporsi kategori lalu berharap rata-ratanya pas adalah cara
    yang pernah gagal 500 kali berturut-turut di proyek sebelumnya.
    Proporsi (0.05, 0.10, 0.20, 0.30, 0.35) secara TEORETIS hanya bisa
    menghasilkan rata-rata 3.80 dan simpangan baku 1.17. Tidak ada seed
    yang bisa menyelamatkannya. Jadi proporsinya yang harus dicari
    secara numerik, bukan datanya yang diulang-ulang.

    KENAPA BUKAN PENCARIAN BEBAS:
    Versi pertama fungsi ini memakai softmax atas vektor bebas sepanjang
    n_points. Cara itu gagal dengan cara yang berbeda. Rata-rata dan
    simpangan hanya DUA kendala sementara proporsinya punya n_points
    dikurangi satu derajat kebebasan, sehingga solusinya tak hingga
    banyak dan optimizer mendarat pada bentuk bergerigi. Pada skala 6
    titik, pencarian bebas menghasilkan proporsi dengan dua puncak dan
    satu kategori bernilai nol. Distribusi seperti itu tidak pernah
    muncul pada survei sungguhan.

    BAGAIMANA:
    Parameterisasi dibatasi menjadi diskretisasi normal dengan dua
    parameter saja, mu dan sigma. Karena bentuk normal tunggal puncak,
    hasil diskretisasinya juga dijamin tunggal puncak.
    """
    kategori = np.arange(1, n_points + 1)

    def statistik(p):
        mean = float((kategori * p).sum())
        sd = float(np.sqrt((((kategori - mean) ** 2) * p).sum()))
        return mean, sd

    def loss(par):
        mu, sigma = float(par[0]), abs(float(par[1])) + 1e-6
        mean, sd = statistik(props_from_normal(mu, sigma, n_points))
        return (mean - target_mean) ** 2 + (sd - target_sd) ** 2

    best = None
    for x0 in [(target_mean, target_sd),
               (target_mean + 0.5, target_sd * 1.25),
               (target_mean - 0.5, target_sd * 0.80)]:
        res = minimize(loss, x0, method="Nelder-Mead",
                       options={"maxiter": 20000, "xatol": 1e-10, "fatol": 1e-14})
        if best is None or res.fun < best.fun:
            best = res

    mu, sigma = float(best.x[0]), abs(float(best.x[1]))
    p = props_from_normal(mu, sigma, n_points)
    mean, sd = statistik(p)

    puncak = int(np.argmax(p)) + 1
    if verbose:
        print(f"  n_points={n_points} target=({target_mean}, {target_sd}) "
              f"-> mean={mean:.4f} sd={sd:.4f} mu={mu:.3f} sigma={sigma:.3f} "
              f"puncak di kategori {puncak}")
    return tuple(np.round(p, 4)), mean, sd


# ==========================================================
# MESIN KONVERSI
# ==========================================================
class OrdinalEngine:

    def __init__(self, seed=None, proportions=None, n_points=None):
        seed = seed if seed is not None else config.RANDOM_SEED + 3
        self.rng = np.random.default_rng(seed)
        self.n_points = n_points or config.SCALE_POINTS
        self.proportions = proportions or config.likert_proportions()
        if len(self.proportions) != self.n_points:
            raise ValueError("Panjang proporsi tidak sama dengan jumlah titik skala.")
        self.base_thresholds = self._build_base_thresholds()

    def _build_base_thresholds(self):
        """
        Proporsi kumulatif diubah jadi batas z-score lewat norm.ppf.
        Angka kumulatif terakhir (1.00) tidak dipakai, karena batas
        terakhir berarti "tak hingga".
        """
        cumulative = np.cumsum(self.proportions)
        return norm.ppf(cumulative[:-1])

    def convert_one_column(self, scores):
        """
        Tiap item diberi pergeseran acak kecil pada ambangnya, supaya
        distribusi antar indikator tidak identik. Di survei sungguhan,
        satu item memang biasanya sedikit lebih mudah disetujui
        dibanding item lain dalam konstruk yang sama.
        """
        jitter = self.rng.normal(0, config.THRESHOLD_JITTER_SD,
                                 size=len(self.base_thresholds))
        thresholds_item = np.sort(self.base_thresholds + jitter)
        kategori = np.searchsorted(thresholds_item, scores) + 1
        return np.clip(kategori, 1, self.n_points)

    def _thresholds_from(self, proportions):
        cumulative = np.cumsum(proportions)
        return norm.ppf(cumulative[:-1])

    def convert_dataframe(self, df_continuous):
        """
        TAMBAHAN PROYEK EV KALBAR.
        Kalau config.CONSTRUCT_LIKERT_TARGET berisi target rerata per
        konstruk, tiap konstruk memakai set ambangnya sendiri. Tanpa ini
        seluruh konstruk terpaksa berbagi satu rerata global, sehingga
        Government Promotion tidak bisa dibuat lebih rendah daripada
        Perceived Usefulness. Padahal justru selisih itulah yang membuat
        datanya masuk akal untuk konteks Kalimantan Barat.
        """
        target = getattr(config, "CONSTRUCT_LIKERT_TARGET", None)
        peta = {}
        if target:
            for konstruk, (m, s) in target.items():
                props, _, _ = find_proportions(self.n_points, m, s)
                thr = self._thresholds_from(props)
                for item in config.CONSTRUCTS[konstruk]:
                    peta[item] = thr

        hasil = df_continuous.copy()
        for kolom in df_continuous.columns:
            simpan = self.base_thresholds
            if kolom in peta:
                self.base_thresholds = peta[kolom]
            hasil[kolom] = self.convert_one_column(df_continuous[kolom].values)
            self.base_thresholds = simpan
        return hasil


if __name__ == "__main__":
    print("=== PENCARIAN PROPORSI NUMERIK ===")
    for n_points, tm, ts in [(4, 3.10, 0.78), (5, 3.90, 0.95),
                             (6, 4.60, 1.10), (7, 5.35, 1.35)]:
        p, m, s = find_proportions(n_points, tm, ts, verbose=True)
        print(f"  proporsi -> {p}  (total {round(sum(p), 4)})\n")

    print("=== CEK KONVERSI ===")
    rng = np.random.default_rng(1)
    z = rng.normal(0, 1, 5000)
    eng = OrdinalEngine(seed=1)
    kat = eng.convert_one_column(z)
    print("  ambang dasar :", np.round(eng.base_thresholds, 3))
    print("  rata-rata    :", round(kat.mean(), 3))
    print("  simpangan    :", round(kat.std(), 3))
    unik, jml = np.unique(kat, return_counts=True)
    print("  sebaran      :", dict(zip(unik.tolist(), np.round(jml / len(kat), 3).tolist())))

    # Bukti pembulatan naif melemahkan korelasi lebih jauh
    a = rng.normal(0, 1, 20000)
    b = 0.70 * a + np.sqrt(1 - 0.70 ** 2) * rng.normal(0, 1, 20000)
    naif_a = np.clip(np.round(a * 0.8 + 4), 1, 5)
    naif_b = np.clip(np.round(b * 0.8 + 4), 1, 5)
    e2 = OrdinalEngine(seed=7)
    thr_a = e2.convert_one_column(a)
    thr_b = e2.convert_one_column(b)
    print("\n=== PEMBULATAN NAIF vs AMBANG BATAS (korelasi asli 0.700) ===")
    print("  pembulatan naif :", round(np.corrcoef(naif_a, naif_b)[0, 1], 4))
    print("  ambang batas    :", round(np.corrcoef(thr_a, thr_b)[0, 1], 4))
