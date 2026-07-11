"""
validate_cfa.py
----------------
Alat pemeriksa TAMBAHAN (bukan bagian dari pipeline generator).
Menguji dataset_sem.csv yang sudah dihasilkan lewat CFA sungguhan
memakai semopy, untuk melihat fit index (CFI, TLI, RMSEA) -- ukuran
yang TIDAK dicek oleh quality_control.py karena nilainya bergantung
pada model SEM yang diuji, bukan cuma pada datanya sendiri.

CATATAN PERBAIKAN:
File ini sekarang JUGA menghitung AVE, CR, dan Discriminant Validity
langsung dari loading dan korelasi antar-konstruk hasil CFA sungguhan
(bukan pendekatan cepat seperti di quality_control.py). Ini penting
karena quality_control.py memakai estimasi cepat (principal
component) supaya bisa dipanggil ratusan kali oleh optimizer.py --
cepat, tapi kadang meleset cukup jauh (pernah selisih 0.08 pada AVE
di kasus nyata). Angka dari file INI yang paling dekat dengan apa
yang akan Anda lihat nanti di AMOS -- jadikan ini pemeriksa akhir
sebelum capek-capek buka AMOS/SPSS.

Cara pakai: pastikan sudah menjalankan main.py dulu (supaya
output/dataset_sem.csv ada), lalu:
    python validate_cfa.py
"""

import numpy as np
import pandas as pd
import semopy
from semopy import calc_stats

import config


def build_model_syntax():
    """
    KENAPA:
    semopy (sama seperti lavaan di R) butuh "resep" model dalam
    bentuk teks, bukan dictionary Python biasa -- ini notasi standar
    di dunia software SEM, supaya bisa dibaca manusia maupun program.

    BAGAIMANA:
    Untuk tiap konstruk, tulis satu baris:
        NAMA_KONSTRUK =~ item1 + item2 + item3 + ...
    Tanda "=~" artinya "diukur oleh" -- konstruk laten di kiri
    diasumsikan menjadi penyebab dari indikator-indikator di kanan.
    Ini persis rumus yang kita pakai waktu MEMBUAT data di
    indicator.py, cuma sekarang arahnya dibalik: sebelumnya kita
    tahu loading lalu membuat data, sekarang semopy diminta MENEBAK
    loadingnya dari data.
    """
    lines = []
    for construct, items in config.CONSTRUCTS.items():
        indicators = " + ".join(items)
        lines.append(f"{construct} =~ {indicators}")
    return "\n".join(lines)


def run_cfa(csv_path):
    """
    KENAPA:
    Ini yang benar-benar menjalankan estimasi CFA -- proses ini yang
    di dunia nyata dilakukan AMOS/SmartPLS/lavaan.

    BAGAIMANA:
    Baca CSV, kasih tahu semopy modelnya lewat build_model_syntax(),
    lalu model.fit(df) mencari nilai loading & korelasi antar
    konstruk yang paling cocok dengan data. calc_stats() menghitung
    fit index dari hasil estimasi tadi.
    """
    df = pd.read_csv(csv_path)
    model_syntax = build_model_syntax()

    model = semopy.Model(model_syntax)
    model.fit(df)

    stats = calc_stats(model)
    return model, stats


def extract_standardized_loadings(model):
    """
    KENAPA:  # <-- BARU
    Untuk menghitung AVE & CR yang BENAR (bukan pendekatan cepat),
    kita butuh loading standardized yang SUNGGUH-SUNGGUH diestimasi
    semopy dari data -- persis angka yang nanti keluar di AMOS pada
    tabel "Standardized Regression Weights".

    BAGAIMANA:
    model.inspect(std_est=True) mengembalikan semua parameter model
    (loading, kovarians, varians) dalam satu tabel. Baris yang kita
    mau adalah yang op == '~' (artinya "item diukur oleh konstruk").
    Hasilnya disusun ulang jadi dictionary: loadings["PE"] = [0.75, 0.81, ...]
    """
    est = model.inspect(std_est=True)
    measurement = est[est["op"] == "~"]

    loadings = {}
    for construct, items in config.CONSTRUCTS.items():
        rows = measurement[measurement["rval"] == construct]
        # urutkan sesuai urutan item di config.py, bukan urutan acak dari semopy
        ordered = rows.set_index("lval").loc[items]
        loadings[construct] = ordered["Est. Std"].astype(float).values
    return loadings


def extract_factor_correlations(model):
    """
    KENAPA:  # <-- BARU
    Discriminant Validity (Fornell-Larcker) butuh korelasi antar
    KONSTRUK LATEN sungguhan -- bukan korelasi antar skor komposit
    (rata-rata item), yang selalu lebih lemah dari korelasi aslinya
    karena mengandung error pengukuran. semopy menghitung korelasi
    laten ini langsung sebagai bagian dari proses CFA, sama seperti
    AMOS.

    BAGAIMANA:
    Baris dengan op == '~~' dan lval != rval adalah kovarians/korelasi
    ANTAR konstruk (baris dengan lval == rval adalah varians konstruk
    itu sendiri, bukan yang kita mau di sini).
    """
    est = model.inspect(std_est=True)
    factor_corr = est[(est["op"] == "~~") & (est["lval"] != est["rval"])]

    corr_dict = {}
    for _, row in factor_corr.iterrows():
        a, b = row["lval"], row["rval"]
        value = float(row["Est. Std"])
        corr_dict[(a, b)] = value
        corr_dict[(b, a)] = value
    return corr_dict


def calculate_cr_ave(loadings_array):
    """
    Rumus Composite Reliability dan AVE standar -- SAMA PERSIS dengan
    _calculate_cr_ave di quality_control.py, cuma sekarang loading
    yang dimasukkan adalah loading CFA sungguhan, bukan estimasi cepat.
    """
    loadings_arr = np.array(loadings_array)
    sum_loading = loadings_arr.sum()
    sum_error = np.sum(1 - loadings_arr ** 2)
    cr = (sum_loading ** 2) / (sum_loading ** 2 + sum_error)
    ave = np.mean(loadings_arr ** 2)
    return cr, ave


def check_validity_and_discriminant(model):
    """
    KENAPA:  # <-- BARU
    Ini fungsi utama yang menggantikan peran "cek AVE/CR/Discriminant
    Validity manual di Excel" pada Langkah 1.4-1.5 tutorial AMOS Anda
    -- bedanya, ini dihitung dari CFA sungguhan (semopy), jadi hasilnya
    seharusnya sangat dekat (biasanya selisih < 0.01) dengan yang nanti
    keluar di AMOS.

    BAGAIMANA:
    1. Ambil loading sungguhan tiap konstruk -> hitung AVE & CR.
    2. Ambil korelasi antar-konstruk sungguhan -> untuk tiap konstruk,
       cari korelasi tertinggi dengan konstruk lain -> bandingkan
       dengan akar(AVE) konstruk itu.
    """
    loadings = extract_standardized_loadings(model)
    factor_corr = extract_factor_correlations(model)

    validity = {}
    for construct, items in config.CONSTRUCTS.items():
        cr, ave = calculate_cr_ave(loadings[construct])
        validity[construct] = {
            "loadings": loadings[construct],
            "cr": cr, "ave": ave,
            "cr_passed": cr >= config.CR_MIN,
            "ave_passed": ave >= config.AVE_MIN,
        }

    discriminant = {}
    for construct in config.CONSTRUCTS:
        sqrt_ave = validity[construct]["ave"] ** 0.5
        other_corrs = {
            other: factor_corr[(construct, other)]
            for other in config.CONSTRUCTS if other != construct
        }
        max_partner = max(other_corrs, key=other_corrs.get)
        max_corr = other_corrs[max_partner]
        discriminant[construct] = {
            "sqrt_ave": sqrt_ave,
            "max_corr": max_corr,
            "max_corr_partner": max_partner,
            "passed": sqrt_ave > max_corr,
        }

    return validity, discriminant


def interpret(value, good, acceptable, higher_is_better=True):
    """Bantuan kecil untuk memberi label OK/LAYAK/KURANG pada tiap fit index."""
    if higher_is_better:
        if value >= good:
            return "SANGAT BAIK"
        elif value >= acceptable:
            return "LAYAK"
        return "KURANG"
    else:
        if value <= good:
            return "SANGAT BAIK"
        elif value <= acceptable:
            return "LAYAK"
        return "KURANG"


if __name__ == "__main__":
    print("=== VALIDASI CFA (semopy) ===")
    print("\nModel yang diuji:")
    print(build_model_syntax())

    model, stats = run_cfa(config.OUTPUT_FILENAME)
    s = stats.loc["Value"]  # ambil baris "Value" saja, isinya semua angka fit index

    print("\n--- Fit index ---")
    print(f"Chi-square       : {s['chi2']:.2f} (df={s['DoF']:.0f}, p={s['chi2 p-value']:.4f})")
    print(f"CFI              : {s['CFI']:.3f}  -> {interpret(s['CFI'], 0.95, 0.90)}")
    print(f"TLI              : {s['TLI']:.3f}  -> {interpret(s['TLI'], 0.95, 0.90)}")
    print(f"RMSEA            : {s['RMSEA']:.3f}  -> "
          f"{interpret(s['RMSEA'], 0.06, 0.08, higher_is_better=False)}")

    print("\nPatokan umum: CFI/TLI >= 0.90 layak, >= 0.95 sangat baik.")
    print("RMSEA <= 0.08 layak, <= 0.06 sangat baik. (Hu & Bentler, 1999)")

    # ==========================================================
    # BAGIAN BARU: AVE, CR, Discriminant Validity dari CFA sungguhan
    # ==========================================================
    validity, discriminant = check_validity_and_discriminant(model)

    print("\n--- AVE & CR (dari loading CFA sungguhan, setara AMOS) ---")
    validity_all_ok = True
    for construct, res in validity.items():
        ok = res["cr_passed"] and res["ave_passed"]
        validity_all_ok = validity_all_ok and ok
        status = "OK" if ok else "GAGAL"
        loadings_str = ", ".join(f"{v:.3f}" for v in res["loadings"])
        print(f"  {construct}: AVE={res['ave']:.3f}, CR={res['cr']:.3f} -> {status}  "
              f"(loading: {loadings_str})")

    print("\n--- Discriminant Validity (Fornell-Larcker), dari korelasi laten sungguhan ---")
    discriminant_all_ok = True
    for construct, res in discriminant.items():
        discriminant_all_ok = discriminant_all_ok and res["passed"]
        status = "OK" if res["passed"] else "GAGAL"
        print(f"  {construct}: sqrt(AVE)={res['sqrt_ave']:.3f} vs korelasi tertinggi="
              f"{res['max_corr']:.3f} (dengan {res['max_corr_partner']}) -> {status}")

    print(f"\n=== KESIMPULAN AKHIR ===")
    print(f"AVE & CR semua konstruk lolos?      {'YA' if validity_all_ok else 'TIDAK'}")
    print(f"Discriminant Validity semua lolos?  {'YA' if discriminant_all_ok else 'TIDAK'}")
    if validity_all_ok and discriminant_all_ok:
        print("-> Model SIAP dibawa ke AMOS, hasilnya seharusnya konsisten dengan ini.")
    else:
        print("-> ADA MASALAH yang perlu diperbaiki SEBELUM ke AMOS -- baca detail di atas.")
        print("   Kalau masalahnya di Discriminant Validity antar 2 konstruk tertentu,")
        print("   biasanya berarti korelasi rancangan antar konstruk itu di config.py")
        print("   (LATENT_CORR_MIN/MAX) perlu diturunkan, atau item-item kedua konstruk")
        print("   itu maknanya memang terlalu berdekatan secara teori.")

    print("\nCatatan: hasil fit index (CFI/TLI/RMSEA) TIDAK dicek otomatis oleh")
    print("quality_control.py karena bergantung pada model SEM, bukan cuma data")
    print("mentahnya. AVE/CR/Discriminant di atas SEKARANG dihitung dari CFA")
    print("sungguhan juga (bukan pendekatan cepat) -- jadikan file ini sebagai")
    print("pemeriksa akhir sebelum ke AMOS/SPSS.")
