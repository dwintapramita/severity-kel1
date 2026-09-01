"""
==========================================================================
 APLIKASI PREDIKSI TINGKAT KEPARAHAN CEDERA (INJURY SEVERITY) - JASA RAHARJA
 Dibangun dengan Streamlit
 --------------------------------------------------------------------------
 Versi ini merupakan gabungan dari dua purwarupa sebelumnya, mengambil
 bagian terbaik dari masing-masing:
   - Label kelas 0/1 yang sudah dikonfirmasi (bukan tebakan)
   - Form submit atomik (tidak rerun di setiap perubahan input)
   - Kartu hasil berwarna sesuai tingkat risiko
   - Perbaikan bug kategori kosong pada jenis_klaim (None vs "None")
   - Beberapa nama file model dicoba otomatis
   - Grafik distribusi probabilitas penuh (kedua kelas)

 CARA MENJALANKAN:
   1. Letakkan file model (mis. "injury_severity_model.pkl") pada folder
      yang sama dengan file ini.
   2. Jika preprocessing TIDAK dibungkus dalam model (file terpisah),
      simpan sebagai 'preprocessor.pkl'; aplikasi otomatis mendeteksinya.
   3. Jalankan: streamlit run app_jasaraharja_final.py
==========================================================================
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ==========================================================================
# 1. KONFIGURASI HALAMAN
# ==========================================================================
st.set_page_config(
    page_title="Prediksi Tingkat Keparahan Cedera | Jasa Raharja",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================================================
# 2. TEMA VISUAL - JASA RAHARJA (Navy & Gold, dengan kartu hasil semantik)
# ==========================================================================
CUSTOM_CSS = """
<style>
    .stApp {
        background: linear-gradient(160deg, #f4f8fc 0%, #ffffff 45%, #eef5fb 100%);
    }
    [data-testid="stHeader"] { background: rgba(255,255,255,.85); }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0B2447 0%, #19376D 100%);
    }
    section[data-testid="stSidebar"] * { color: #ffffff !important; }
    section[data-testid="stSidebar"] hr { border-color: #F2A900; }

    /* ---------- Hero header ---------- */
    .hero {
        padding: 1.9rem 2.2rem;
        border-radius: 20px;
        background: linear-gradient(120deg, #0B2447 0%, #19376D 100%);
        color: white;
        box-shadow: 0 12px 30px rgba(11,36,71,.22);
        margin-bottom: 1.3rem;
        border-left: 8px solid #F2A900;
    }
    .hero h1 { margin: 0; font-size: 1.9rem; color: white; font-weight: 800; }
    .hero p { margin: .5rem 0 0; color: #FFD873; opacity: .95; }

    /* ---------- Section card ---------- */
    .jr-card {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 1.5rem 1.7rem;
        box-shadow: 0 3px 14px rgba(11,36,71,.08);
        border: 1px solid #E4E9F0;
        margin-bottom: 1.3rem;
    }
    .section-label {
        font-weight: 700; color: #0B2447; font-size: 1.05rem;
        border-bottom: 3px solid #F2A900; display: inline-block;
        padding-bottom: 4px; margin-bottom: 1rem;
    }

    /* ---------- Result cards (semantik: aman vs risiko) ---------- */
    .result-card {
        padding: 1.5rem 1.7rem; border-radius: 16px; margin-top: .3rem;
        box-shadow: 0 8px 22px rgba(8,55,91,.10);
    }
    .result-safe { background: #ecfdf5; border: 1px solid #bbf7d0; border-left: 7px solid #16a34a; }
    .result-risk { background: #fef2f2; border: 1px solid #fecaca; border-left: 7px solid #dc2626; }
    .result-card .eyebrow {
        font-size: .78rem; letter-spacing: 1.5px; text-transform: uppercase;
        font-weight: 700; opacity: .75; margin-bottom: .3rem;
    }
    .result-safe .eyebrow { color: #15803d; }
    .result-risk .eyebrow { color: #b91c1c; }
    .result-card h2 { margin: 0 0 .4rem; color: #0B2447; font-size: 1.5rem; }
    .result-card p { margin: 0 0 .3rem; color: #334155; }
    .small-note { font-size: .82rem; color: #64748b; margin-top: .6rem; }

    /* ---------- Buttons ---------- */
    div[data-testid="stFormSubmitButton"] > button {
        width: 100%; border-radius: 10px; border: 0; font-weight: 700;
        background: #F2A900; color: #0B2447; min-height: 3rem; font-size: 1rem;
        transition: .2s;
    }
    div[data-testid="stFormSubmitButton"] > button:hover {
        background: #FFD873; color: #0B2447; transform: translateY(-1px);
    }

    /* ---------- Footer ---------- */
    .jr-footer {
        text-align: center; color: #7C8DA6; font-size: 12.5px;
        margin-top: 2rem; padding-top: .9rem; border-top: 1px solid #DCE3EC;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ==========================================================================
# 3. KONSTANTA & BATASAN FITUR (sesuai X.describe() saat training)
# ==========================================================================
FEATURE_LIMITS_NUMERIK = {
    "usia": {"min": 7, "max": 74, "default": 30},
    "usia_kendaraan_tahun": {"min": 0, "max": 24, "default": 5},
    "jumlah_kendaraan_terlibat": {"min": 0, "max": 4, "default": 1},
}

DAFTAR_PROVINSI = sorted([
    "Lampung", "Sumatera Barat", "Jawa Tengah", "Bangka Belitung", "Kalimantan Utara",
    "DKI Jakarta", "Jawa Barat", "Bali", "Banten", "Papua Barat", "Aceh",
    "Sumatera Utara", "Riau", "Kepulauan Riau", "Jambi", "Sumatera Selatan",
    "Bengkulu", "Yogyakarta", "DI Yogyakarta", "Jawa Timur", "Nusa Tenggara Barat",
    "Nusa Tenggara Timur", "Kalimantan Barat", "Kalimantan Tengah", "Kalimantan Selatan",
    "Kalimantan Timur", "Sulawesi Utara", "Sulawesi Tengah", "Sulawesi Selatan",
    "Sulawesi Tenggara", "Gorontalo", "Sulawesi Barat", "Maluku", "Maluku Utara", "Papua",
])

DAFTAR_GENDER = ["Laki-laki", "Perempuan"]
DAFTAR_JENIS_KECELAKAAN = ["Lalu Lintas Jalan", "Penumpang Angkutan Umum", "Other"]
DAFTAR_JENIS_KENDARAAN = [
    "Sepeda Motor", "Mobil Penumpang", "Angkutan Umum/Bus", "Lainnya",
    "Other", "Truk/Angkutan Barang",
]

# Representasi kategori kosong pada jenis_klaim. Saat training, kategori ini
# tersimpan sebagai nilai kosong (None/NaN), BUKAN string "None" — jika
# tertukar, OneHotEncoder(handle_unknown='ignore') akan menganggapnya
# kategori tak dikenal dan hasil prediksi jadi tidak akurat.
LABEL_TAMPILAN_KLAIM_KOSONG = "Tidak Ada / Tidak Diketahui"
DAFTAR_JENIS_KLAIM = [
    "Lalu Lintas Jalan", "Penumpang Angkutan Umum", "Lainnya",
    LABEL_TAMPILAN_KLAIM_KOSONG,
]

# Beberapa kategori kategorikal punya dua istilah yang mirip ("Lainnya" vs
# "Other"); format_func ini memperjelas maknanya di dropdown tanpa mengubah
# nilai asli yang dikirim ke model.
def format_kategori(value: str) -> str:
    if value == "Other":
        return "Lainnya (Other)"
    if value == LABEL_TAMPILAN_KLAIM_KOSONG:
        return LABEL_TAMPILAN_KLAIM_KOSONG
    return value

# Nama file model - beberapa kemungkinan dicoba otomatis karena beberapa
# platform mengganti spasi/tanda kurung menjadi underscore saat upload.
KEMUNGKINAN_NAMA_FILE_MODEL = [
    "injury_severity_model.pkl",
    "injury_severity_model (2).pkl",
    "injury_severity_model_(2).pkl",
    "injury_severity_model__2_.pkl",
]
NAMA_KOLOM_TARGET = "injury_severity"

# Label kelas target - dikonfirmasi dari definisi target saat training:
# kelas 0 = korban tidak meninggal dunia (luka ringan/berat),
# kelas 1 = korban meninggal dunia.
LABEL_KELAS = {
    0: "Tidak Meninggal Dunia (Luka Ringan/Berat)",
    1: "Meninggal Dunia",
}
KELAS_FATAL = 1  # nilai kelas yang dianggap "risiko tinggi" untuk kartu hasil


# ==========================================================================
# 4. MEMUAT MODEL (dengan cache agar tidak reload setiap interaksi)
# ==========================================================================
@st.cache_resource(show_spinner="Memuat model...")
def load_artifacts():
    """
    Memuat model (dan preprocessor terpisah jika ada) dari disk.
    Model diasumsikan berupa sklearn Pipeline yang sudah menyertakan
    ColumnTransformer (imputer+scaler untuk numerik, imputer+OHE untuk
    kategorikal) di dalamnya, sehingga preprocessor terpisah bersifat opsional.
    """
    base_dir = Path(__file__).resolve().parent
    preprocessor_path = base_dir / "preprocessor.pkl"

    model = None
    model_ditemukan = None
    for nama_file in KEMUNGKINAN_NAMA_FILE_MODEL:
        kandidat = base_dir / nama_file
        if kandidat.exists():
            model_ditemukan = kandidat
            break

    if model_ditemukan is not None:
        model = joblib.load(model_ditemukan)

    preprocessor = joblib.load(preprocessor_path) if preprocessor_path.exists() else None
    return model, preprocessor


def build_input_dataframe(usia, usia_kendaraan, jml_kendaraan,
                           provinsi, gender, jenis_kecelakaan,
                           jenis_kendaraan, jenis_klaim):
    """Menyusun satu baris DataFrame dari input pengguna."""
    jenis_klaim_final = (
        None if jenis_klaim == LABEL_TAMPILAN_KLAIM_KOSONG else jenis_klaim
    )
    data = {
        "usia": [usia],
        "usia_kendaraan_tahun": [usia_kendaraan],
        "jumlah_kendaraan_terlibat": [jml_kendaraan],
        "provinsi": [provinsi],
        "gender": [gender],
        "jenis_kecelakaan": [jenis_kecelakaan],
        "jenis_kendaraan": [jenis_kendaraan],
        "jenis_klaim": [jenis_klaim_final],
    }
    return pd.DataFrame(data)


def probabilitas_kelas_fatal(model, data: pd.DataFrame):
    """Ambil probabilitas kelas 'risiko tinggi' (fatal) secara aman."""
    if not hasattr(model, "predict_proba"):
        return None
    proba = model.predict_proba(data)[0]
    classes = list(getattr(model, "classes_", range(len(proba))))
    if KELAS_FATAL in classes:
        return float(proba[classes.index(KELAS_FATAL)])
    return float(proba[-1]) if len(proba) == 2 else None


# ==========================================================================
# 5. SIDEBAR - INFORMASI APLIKASI
# ==========================================================================
with st.sidebar:
    st.markdown("## 🛡️ Jasa Raharja")
    st.markdown("**Hadir Melindungi Bangsa**")
    st.markdown("---")
    st.markdown("### Tentang Aplikasi")
    st.write(
        "Purwarupa (prototype) untuk membantu estimasi awal **tingkat "
        "keparahan cedera (injury severity)** berdasarkan karakteristik "
        "korban, kendaraan, dan kejadian kecelakaan."
    )
    st.markdown("---")
    st.markdown("### Arti Kelas Prediksi")
    st.info(f"**Kelas 0:** {LABEL_KELAS[0]}")
    st.error(f"**Kelas 1:** {LABEL_KELAS[1]}")
    st.markdown("---")
    st.markdown("### Batasan Input Fitur")
    st.markdown(
        f"""
        - Usia korban: {FEATURE_LIMITS_NUMERIK['usia']['min']}–{FEATURE_LIMITS_NUMERIK['usia']['max']} tahun
        - Usia kendaraan: {FEATURE_LIMITS_NUMERIK['usia_kendaraan_tahun']['min']}–{FEATURE_LIMITS_NUMERIK['usia_kendaraan_tahun']['max']} tahun
        - Kendaraan terlibat: {FEATURE_LIMITS_NUMERIK['jumlah_kendaraan_terlibat']['min']}–{FEATURE_LIMITS_NUMERIK['jumlah_kendaraan_terlibat']['max']}
        """
    )
    st.markdown("---")
    st.caption("Alat bantu analitik internal • Bukan keputusan medis atau penetapan santunan resmi")
    st.caption("Data Management Team - Jasa Raharja")

# ==========================================================================
# 6. HEADER UTAMA
# ==========================================================================
st.markdown(
    """
    <div class="hero">
        <h1>🛡️ Sistem Prediksi Tingkat Keparahan Cedera</h1>
        <p>PT Jasa Raharja (Persero) — Decision Support System berbasis machine learning</p>
    </div>
    """,
    unsafe_allow_html=True,
)

model, preprocessor = load_artifacts()

if model is None:
    st.warning(
        "⚠️ File model belum ditemukan pada direktori aplikasi "
        f"(dicari: {', '.join(KEMUNGKINAN_NAMA_FILE_MODEL)}). "
        "Form input di bawah tetap dapat dicoba, namun prediksi tidak akan "
        "berjalan sampai model diletakkan pada folder yang sama dengan file ini."
    )

# ==========================================================================
# 7. FORM INPUT FITUR (atomik - hanya diproses saat tombol ditekan)
# ==========================================================================
with st.form("prediction_form", clear_on_submit=False):
    st.markdown('<div class="jr-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">📋 Data Korban &amp; Kendaraan</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        usia = st.number_input(
            "Usia Korban (tahun)",
            min_value=FEATURE_LIMITS_NUMERIK["usia"]["min"],
            max_value=FEATURE_LIMITS_NUMERIK["usia"]["max"],
            value=FEATURE_LIMITS_NUMERIK["usia"]["default"],
            step=1,
        )
    with c2:
        usia_kendaraan = st.number_input(
            "Usia Kendaraan (tahun)",
            min_value=FEATURE_LIMITS_NUMERIK["usia_kendaraan_tahun"]["min"],
            max_value=FEATURE_LIMITS_NUMERIK["usia_kendaraan_tahun"]["max"],
            value=FEATURE_LIMITS_NUMERIK["usia_kendaraan_tahun"]["default"],
            step=1,
        )
    with c3:
        jml_kendaraan = st.number_input(
            "Jumlah Kendaraan Terlibat",
            min_value=FEATURE_LIMITS_NUMERIK["jumlah_kendaraan_terlibat"]["min"],
            max_value=FEATURE_LIMITS_NUMERIK["jumlah_kendaraan_terlibat"]["max"],
            value=FEATURE_LIMITS_NUMERIK["jumlah_kendaraan_terlibat"]["default"],
            step=1,
        )

    gender = st.selectbox("Jenis Kelamin", DAFTAR_GENDER)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="jr-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">🚗 Informasi Kejadian &amp; Klaim</div>', unsafe_allow_html=True)

    c4, c5 = st.columns(2)
    with c4:
        provinsi = st.selectbox("Provinsi Kejadian", DAFTAR_PROVINSI, index=DAFTAR_PROVINSI.index("DKI Jakarta"))
        jenis_kecelakaan = st.selectbox(
            "Jenis Kecelakaan", DAFTAR_JENIS_KECELAKAAN, format_func=format_kategori
        )
    with c5:
        jenis_kendaraan = st.selectbox(
            "Jenis Kendaraan", DAFTAR_JENIS_KENDARAAN, format_func=format_kategori
        )
        jenis_klaim = st.selectbox(
            "Jenis Klaim", DAFTAR_JENIS_KLAIM, format_func=format_kategori
        )

    st.markdown("</div>", unsafe_allow_html=True)

    submitted = st.form_submit_button("🔍 Analisis Tingkat Keparahan", type="primary")

# ==========================================================================
# 8. HASIL PREDIKSI
# ==========================================================================
if submitted:
    input_df = build_input_dataframe(
        usia, usia_kendaraan, jml_kendaraan,
        provinsi, gender, jenis_kecelakaan, jenis_kendaraan, jenis_klaim,
    )

    st.markdown('<div class="jr-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">🔍 Hasil Prediksi</div>', unsafe_allow_html=True)

    if model is None:
        st.error(
            "Model belum tersedia. Letakkan salah satu file berikut pada "
            f"direktori aplikasi ini lalu jalankan ulang: {', '.join(KEMUNGKINAN_NAMA_FILE_MODEL)}."
        )
    else:
        try:
            data_final = input_df
            if preprocessor is not None:
                data_final = preprocessor.transform(input_df)

            prediksi = model.predict(data_final)[0]
            kelas_model = list(getattr(model, "classes_", []))
            label_prediksi = LABEL_KELAS.get(prediksi, f"Kelas {prediksi}")
            proba_fatal = probabilitas_kelas_fatal(model, data_final)

            is_risiko_tinggi = prediksi == KELAS_FATAL
            card_class = "result-risk" if is_risiko_tinggi else "result-safe"
            eyebrow = "Risiko Tinggi" if is_risiko_tinggi else "Risiko Rendah"

            probabilitas_html = ""
            if proba_fatal is not None:
                probabilitas_html = (
                    f"<p><b>Probabilitas kelas Meninggal Dunia:</b> {proba_fatal:.1%}</p>"
                )

            st.markdown(
                f"""
                <div class="result-card {card_class}">
                    <div class="eyebrow">{eyebrow} — Prediksi {NAMA_KOLOM_TARGET.replace('_', ' ').title()}</div>
                    <h2>{label_prediksi}</h2>
                    {probabilitas_html}
                    <div class="small-note">
                        Hasil merupakan estimasi statistik dan harus divalidasi bersama
                        data serta pertimbangan petugas berwenang.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if hasattr(model, "predict_proba") and kelas_model:
                proba = model.predict_proba(data_final)[0]
                proba_df = pd.DataFrame({
                    "Tingkat Keparahan": [LABEL_KELAS.get(k, f"Kelas {k}") for k in kelas_model],
                    "Probabilitas": np.round(proba, 4),
                }).sort_values("Probabilitas", ascending=False)

                st.markdown("#### Distribusi Probabilitas")
                st.bar_chart(proba_df.set_index("Tingkat Keparahan"))
                st.dataframe(proba_df, use_container_width=True, hide_index=True)

            with st.expander("Lihat data input yang dikirim ke model"):
                st.dataframe(input_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
            st.info(
                "Periksa kembali kesesuaian nama kolom, urutan fitur, dan "
                "encoding kategori antara form input ini dengan proses training model. "
                "Jika model bukan Pipeline lengkap (preprocessing terpisah), "
                "pastikan file 'preprocessor.pkl' tersedia."
            )

    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================================================
# 9. FOOTER
# ==========================================================================
st.markdown(
    """
    <div class="jr-footer">
        Aplikasi ini merupakan alat bantu estimasi internal dan tidak menggantikan
        proses verifikasi/asesmen resmi tingkat keparahan cedera oleh Jasa Raharja.<br>
        © 2026 PT Jasa Raharja (Persero) — Data Management Team
    </div>
    """,
    unsafe_allow_html=True,
)
