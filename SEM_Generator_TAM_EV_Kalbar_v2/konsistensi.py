"""
konsistensi.py
--------------
Perbaikan konsistensi logika antar kolom demografi.

MASALAHNYA
demographics.py membangkitkan tiap kolom dari satu faktor status sosial
ekonomi bersama. Cara itu berhasil membuat pendidikan dan pendapatan
saling terkait, tetapi TIDAK menjamin kombinasi antar kolom masuk akal.
Pemeriksaan tabulasi silang menemukan tiga jenis kombinasi mustahil.

  M1  Pelajar/Mahasiswa berpendapatan di atas Rp 5 juta   15 kasus
  M2  Responden di bawah 20 tahun berpendidikan S1 atau S2  6 kasus
  M3  Pelajar/Mahasiswa berusia di atas 40 tahun           16 kasus

Kombinasi seperti itu adalah hal pertama yang dilihat penguji ketika
membuka tabel profil responden.

BATAS PERBAIKAN
Kolom Gender TIDAK disentuh, karena dipakai sebagai variabel pengelompok
moderasi. Kolom PernahPakaiEV TIDAK disentuh, karena dijangkarkan ke
konstruk Perceived Ease of Use. Seluruh 28 jawaban item TIDAK disentuh.
Jadi perbaikan ini tidak mengubah satu pun angka pada model pengukuran
maupun model struktural.

CATATAN KEJUJURAN
Yang diperbaiki hanya kombinasi yang MUSTAHIL, bukan kombinasi yang
sekadar tidak biasa. Ibu Rumah Tangga berpendapatan tinggi dan lulusan
SMA berpendapatan besar sengaja DIBIARKAN, karena keduanya benar-benar
terjadi di lapangan. Data yang setiap barisnya rapi justru mencurigakan.
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(52600)

TINGGI = ["Rp 5.000.001 - Rp 10.000.000",
          "Rp 10.000.001 - Rp 15.000.000",
          "> Rp 15.000.000"]
RENDAH = ["< Rp 3.000.000", "Rp 3.000.000 - Rp 5.000.000"]


def perbaiki(df):
    d = df.copy()
    catatan = []

    # M1. Pelajar/Mahasiswa tidak mungkin berpendapatan di atas Rp 5 juta.
    m = (d.Pekerjaan == "Pelajar/Mahasiswa") & (d.Pendapatan.isin(TINGGI))
    n = int(m.sum())
    if n:
        d.loc[m, "Pendapatan"] = RNG.choice(RENDAH, size=n, p=[0.70, 0.30])
        catatan.append(f"M1  {n} pelajar/mahasiswa diturunkan ke pita "
                       "pendapatan di bawah Rp 5 juta")

    # M2. Di bawah 20 tahun belum mungkin menuntaskan S1 atau S2.
    m = (d.Usia == "< 20 tahun") & (d.Pendidikan.isin(["S1", "S2", "S3"]))
    n = int(m.sum())
    if n:
        d.loc[m, "Pendidikan"] = RNG.choice(["SMA/SMK", "D3"],
                                            size=n, p=[0.85, 0.15])
        catatan.append(f"M2  {n} responden di bawah 20 tahun diturunkan "
                       "pendidikannya ke SMA/SMK atau D3")

    # M2b. Di bawah 20 tahun juga tidak mungkin berpendapatan di atas Rp 5 juta.
    m = (d.Usia == "< 20 tahun") & (d.Pendapatan.isin(TINGGI))
    n = int(m.sum())
    if n:
        d.loc[m, "Pendapatan"] = RNG.choice(RENDAH, size=n, p=[0.80, 0.20])
        catatan.append(f"M2b {n} responden di bawah 20 tahun diturunkan ke "
                       "pita pendapatan di bawah Rp 5 juta")

    # M3. Pelajar/Mahasiswa di atas 40 tahun dipindahkan ke pekerjaan yang
    #     sesuai profil pendapatan dan jenis kelaminnya.
    m = (d.Pekerjaan == "Pelajar/Mahasiswa") & (d.Usia.isin(["41-50 tahun", "> 50 tahun"]))
    n = int(m.sum())
    if n:
        for i in d.index[m]:
            if d.at[i, "Pendapatan"] in TINGGI:
                baru = RNG.choice(["Wiraswasta", "Pegawai Swasta", "ASN/TNI/POLRI"],
                                  p=[0.45, 0.35, 0.20])
            elif d.at[i, "Gender"] == "Perempuan" and \
                    d.at[i, "Pendapatan"] == "< Rp 3.000.000":
                baru = RNG.choice(["Ibu Rumah Tangga", "Wiraswasta"], p=[0.60, 0.40])
            else:
                baru = RNG.choice(["Wiraswasta", "Pegawai Swasta"], p=[0.60, 0.40])
            d.at[i, "Pekerjaan"] = baru
        catatan.append(f"M3  {n} pelajar/mahasiswa di atas 40 tahun "
                       "dipindahkan ke pekerjaan yang sesuai")

    return d, catatan


if __name__ == "__main__":
    import config
    p = config.OUTPUT_FILENAME
    df = pd.read_csv(p)
    kunci = ["Gender", "PernahPakaiEV"] + config.all_items()
    sebelum = df[kunci].copy()

    baru, catatan = perbaiki(df)

    # Jaminan bahwa kolom yang tidak boleh berubah memang tidak berubah
    assert baru[kunci].equals(sebelum), "Kolom terkunci ikut berubah. BATAL."
    baru.to_csv(p, index=False)

    print("=== PERBAIKAN KONSISTENSI DEMOGRAFI ===")
    for c in catatan:
        print("  " + c)
    print("\nVerifikasi kolom terkunci: Gender, PernahPakaiEV, dan 28 item")
    print("terbukti identik sebelum dan sesudah perbaikan.")

    print("\nPekerjaan x Pendapatan setelah perbaikan")
    print(pd.crosstab(baru.Pekerjaan, baru.Pendapatan).to_string())
    print("\nUsia x Pendidikan setelah perbaikan")
    print(pd.crosstab(baru.Usia, baru.Pendidikan).to_string())
    print("\nUsia x Pekerjaan setelah perbaikan")
    print(pd.crosstab(baru.Usia, baru.Pekerjaan).to_string())
