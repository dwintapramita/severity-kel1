"""
==========================================================================
 APLIKASI PREDIKSI TINGKAT KEPARAHAN CEDERA (INJURY SEVERITY) - JASA RAHARJA
 Dibangun dengan Streamlit
 --------------------------------------------------------------------------
 CARA MENJALANKAN:
   1. letakkan file model "injury_severity_model (2).pkl" pada folder yang
      sama dengan file ini. Model diasumsikan berupa sklearn Pipeline yang
      SUDAH menyertakan ColumnTransformer (SimpleImputer+RobustScaler untuk
      kolom numerik, SimpleImputer+OneHotEncoder untuk kolom kategorikal),
      sehingga cukup memanggil model.predict() pada DataFrame mentah dengan
      nama kolom yang sesuai.
   2. jika preprocessing TIDAK dibungkus dalam model (disimpan file
      terpisah), simpan sebagai 'preprocessor.pkl' pada folder yang sama;
      aplikasi ini akan otomatis mendeteksi dan menggunakannya sebelum
      memanggil model.predict().
   3. pastikan nama & urutan kolom pada `build_input_dataframe()` PERSIS
      sama dengan kolom X saat training.
   4. jalankan melalui terminal:
        streamlit run app_jasaraharja.py
==========================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime

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
# 2. TEMA VISUAL - JASA RAHARJA (Navy & Gold)
# ==========================================================================
PRIMARY_NAVY = "#0B2447"
SECONDARY_NAVY = "#19376D"
ACCENT_GOLD = "#F2A900"
ACCENT_GOLD_LIGHT = "#FFD873"
BG_LIGHT = "#F5F7FA"
TEXT_LIGHT = "#FFFFFF"

CUSTOM_CSS = f"""
<style>
    /* ---------- Global ---------- */
    .stApp {{
        background-color: {BG_LIGHT};
    }}

    /* ---------- Header Banner ---------- */
    .jr-header {{
        background: linear-gradient(135deg, {PRIMARY_NAVY} 0%, {SECONDARY_NAVY} 100%);
        padding: 28px 36px;
        border-radius: 14px;
        margin-bottom: 28px;
        box-shadow: 0 4px 18px rgba(11, 36, 71, 0.25);
        border-left: 8px solid {ACCENT_GOLD};
    }}
    .jr-header h1 {{
        color: {TEXT_LIGHT};
        font-size: 30px;
        font-weight: 800;
        margin: 0;
        letter-spacing: 0.3px;
    }}
    .jr-header p {{
        color: {ACCENT_GOLD_LIGHT};
        font-size: 15px;
        margin-top: 6px;
        margin-bottom: 0;
    }}

    /* ---------- Section Card ---------- */
    .jr-card {{
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 22px 26px;
        box-shadow: 0 2px 10px rgba(11, 36, 71, 0.08);
        border: 1px solid #E4E9F0;
        margin-bottom: 22px;
    }}
    .jr-card h3 {{
        color: {PRIMARY_NAVY};
        font-size: 18px;
        font-weight: 700;
        border-bottom: 3px solid {ACCENT_GOLD};
        display: inline-block;
        padding-bottom: 4px;
        margin-bottom: 18px;
    }}

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {{
        background-color: {PRIMARY_NAVY};
    }}
    section[data-testid="stSidebar"] * {{
        color: {TEXT_LIGHT} !important;
    }}
    section[data-testid="stSidebar"] hr {{
        border-color: {ACCENT_GOLD};
    }}

    /* ---------- Buttons ---------- */
    div.stButton > button {{
        background-color: {ACCENT_GOLD};
        color: {PRIMARY_NAVY};
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 10px 26px;
        font-size: 15px;
        transition: 0.2s;
        width: 100%;
    }}
    div.stButton > button:hover {{
        background-color: {ACCENT_GOLD_LIGHT};
        color: {PRIMARY_NAVY};
        transform: translateY(-1px);
    }}

    /* ---------- Result Box ---------- */
    .jr-result {{
        background: linear-gradient(135deg, {SECONDARY_NAVY} 0%, {PRIMARY_NAVY} 100%);
        color: {TEXT_LIGHT};
        padding: 26px 30px;
        border-radius: 12px;
        text-align: center;
        border: 2px solid {ACCENT_GOLD};
        margin-top: 10px;
    }}
    .jr-result .label {{
        font-size: 14px;
        color: {ACCENT_GOLD_LIGHT};
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 6px;
    }}
    .jr-result .value {{
        font-size: 30px;
        font-weight: 800;
    }}

    /* ---------- Footer ---------- */
    .jr-footer {{
        text-align: center;
        color: #7C8DA6;
        font-size: 12.5px;
        margin-top: 34px;
        padding-top: 14px;
        border-top: 1px solid #DCE3EC;
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ==========================================================================
# 3. BATASAN FITUR (sesuai hasil eksplorasi data / X.describe())
#    -> Sesuaikan nilai ini apabila rentang data pada training berbeda.
# ==========================================================================
FEATURE_LIMITS_NUMERIK = {
    "usia": {"min": 7, "max": 74, "default": 30},
    "usia_kendaraan_tahun": {"min": 0, "max": 24, "default": 5},
    "jumlah_kendaraan_terlibat": {"min": 0, "max": 4, "default": 1},
}

DAFTAR_PROVINSI = [
    "Lampung", "Sumatera Barat", "Jawa Tengah", "Bangka Belitung", "Kalimantan Utara",
    "DKI Jakarta", "Jawa Barat", "Bali", "Banten", "Papua Barat", "Aceh",
    "Sumatera Utara", "Riau", "Kepulauan Riau", "Jambi", "Sumatera Selatan",
    "Bengkulu", "Yogyakarta", "DI Yogyakarta", "Jawa Timur", "Nusa Tenggara Barat",
    "Nusa Tenggara Timur", "Kalimantan Barat", "Kalimantan Tengah", "Kalimantan Selatan",
    "Kalimantan Timur", "Sulawesi Utara", "Sulawesi Tengah", "Sulawesi Selatan",
    "Sulawesi Tenggara", "Gorontalo", "Sulawesi Barat", "Maluku", "Maluku Utara", "Papua",
]

DAFTAR_GENDER = ["Laki-laki", "Perempuan"]
DAFTAR_JENIS_KECELAKAAN = ["Lalu Lintas Jalan", "Penumpang Angkutan Umum", "Other"]
DAFTAR_JENIS_KENDARAAN = [
    "Sepeda Motor", "Mobil Penumpang", "Angkutan Umum/Bus", "Lainnya",
    "Other", "Truk/Angkutan Barang",
]

# Representasi eksplisit untuk kategori kosong/tidak ada pada jenis_klaim.
# Saat training, kategori ini tersimpan sebagai nilai kosong (None/NaN),
# BUKAN string "None". Jika keduanya tertukar, OneHotEncoder akan
# menganggapnya kategori tak dikenal (handle_unknown='ignore') dan
# hasil prediksi menjadi tidak akurat.
# HARUS didefinisikan sebelum DAFTAR_JENIS_KLAIM karena dipakai di sana.
LABEL_TAMPILAN_KLAIM_KOSONG = "Tidak Ada / None"

DAFTAR_JENIS_KLAIM = [
    "Lalu Lintas Jalan", "Penumpang Angkutan Umum", "Lainnya",
    LABEL_TAMPILAN_KLAIM_KOSONG,
]

# Nama file model & label kolom target - SESUAIKAN jika berbeda
# (mendukung beberapa kemungkinan nama file, karena beberapa platform
#  otomatis mengganti spasi/tanda kurung menjadi underscore saat upload)
KEMUNGKINAN_NAMA_FILE_MODEL = [
    "injury_severity_model (2).pkl",
    "injury_severity_model_(2).pkl",
    "injury_severity_model__2_.pkl",
    "injury_severity_model.pkl",
]
NAMA_KOLOM_TARGET = "injury_severity"

# PENTING: model ini hasil klasifikasi BINER (model.classes_ -> [0, 1]).
# Model tidak menyimpan label teks untuk tiap kelas, jadi pemetaan di bawah
# ini HARUS disesuaikan secara manual agar sesuai definisi label saat
# training (misalnya dari `y.unique()` atau `LabelEncoder.classes_` pada
# notebook training). Nilai di bawah hanyalah PLACEHOLDER.
LABEL_KELAS = {
    0: "Kelas 0 (mis. Tidak Berat)",
    1: "Kelas 1 (mis. Berat)",
}


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
    Mengembalikan tuple (model, preprocessor_or_None).
    """
    preprocessor_path = Path("preprocessor.pkl")

    model = None
    preprocessor = None
    model_path_ditemukan = None

    for nama_file in KEMUNGKINAN_NAMA_FILE_MODEL:
        kandidat = Path(nama_file)
        if kandidat.exists():
            model_path_ditemukan = kandidat
            break

    if model_path_ditemukan is not None:
        model = joblib.load(model_path_ditemukan)
    if preprocessor_path.exists():
        preprocessor = joblib.load(preprocessor_path)

    return model, preprocessor


def build_input_dataframe(usia, usia_kendaraan, jml_kendaraan,
                           provinsi, gender, jenis_kecelakaan,
                           jenis_kendaraan, jenis_klaim):
    """
    Menyusun satu baris DataFrame dari input pengguna.
    PENTING: urutan & nama kolom harus PERSIS sama dengan data X saat training
    (3 kolom numerik + 5 kolom kategorikal, termasuk jenis_klaim sebagai fitur).
    Sesuaikan urutan/nama kolom di bawah bila berbeda dari X asli.
    """
    # Konversi label tampilan "Tidak Ada / None" kembali menjadi nilai
    # kosong (None) yang PERSIS sama dengan representasi kategori kosong
    # saat training. Ini krusial karena OneHotEncoder membedakan string
    # "None" dari nilai kosong (None/NaN) sebagai dua kategori berbeda.
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


# ==========================================================================
# 5. SIDEBAR - INFORMASI APLIKASI
# ==========================================================================
with st.sidebar:
    st.markdown("## 🛡️ Jasa Raharja")
    st.markdown("**Sistem Prediksi Tingkat Keparahan Cedera**")
    st.markdown("---")
    st.markdown(
        """
        Aplikasi ini merupakan purwarupa (prototype) untuk membantu
        proses estimasi awal **tingkat keparahan cedera (injury severity)**
        berdasarkan karakteristik data kecelakaan yang diinput.
        """
    )
    st.markdown("---")
    st.markdown("**Batasan Input Fitur**")
    st.markdown(
        f"""
        - Usia korban: {FEATURE_LIMITS_NUMERIK['usia']['min']}–{FEATURE_LIMITS_NUMERIK['usia']['max']} tahun
        - Usia kendaraan: {FEATURE_LIMITS_NUMERIK['usia_kendaraan_tahun']['min']}–{FEATURE_LIMITS_NUMERIK['usia_kendaraan_tahun']['max']} tahun
        - Kendaraan terlibat: {FEATURE_LIMITS_NUMERIK['jumlah_kendaraan_terlibat']['min']}–{FEATURE_LIMITS_NUMERIK['jumlah_kendaraan_terlibat']['max']}
        - 5 fitur kategorikal (provinsi, gender, jenis kecelakaan,
          jenis kendaraan, jenis klaim) dibatasi sesuai kategori hasil
          eksplorasi data (`X.describe(exclude='number')`)
        """
    )
    st.markdown("---")
    st.caption(f"Versi purwarupa • {datetime.now().strftime('%Y')}")
    st.caption("Data Management Team - Jasa Raharja")

# ==========================================================================
# 6. HEADER UTAMA
# ==========================================================================
st.markdown(
    """
    <div class="jr-header">
        <h1>🛡️ Sistem Prediksi Tingkat Keparahan Cedera</h1>
        <p>PT Jasa Raharja (Persero) — Data Management &amp; Analytics</p>
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
# 7. FORM INPUT FITUR
# ==========================================================================
col_kiri, col_kanan = st.columns(2, gap="large")

with col_kiri:
    st.markdown('<div class="jr-card">', unsafe_allow_html=True)
    st.markdown("### 📋 Data Korban")

    usia = st.slider(
        "Usia (tahun)",
        min_value=FEATURE_LIMITS_NUMERIK["usia"]["min"],
        max_value=FEATURE_LIMITS_NUMERIK["usia"]["max"],
        value=FEATURE_LIMITS_NUMERIK["usia"]["default"],
    )

    gender = st.selectbox("Jenis Kelamin", DAFTAR_GENDER)

    provinsi = st.selectbox("Provinsi Kejadian", sorted(DAFTAR_PROVINSI))

    st.markdown("</div>", unsafe_allow_html=True)

with col_kanan:
    st.markdown('<div class="jr-card">', unsafe_allow_html=True)
    st.markdown("### 🚗 Data Kecelakaan &amp; Kendaraan")

    jenis_kecelakaan = st.selectbox("Jenis Kecelakaan", DAFTAR_JENIS_KECELAKAAN)

    jenis_kendaraan = st.selectbox("Jenis Kendaraan", DAFTAR_JENIS_KENDARAAN)

    jenis_klaim = st.selectbox("Jenis Klaim", DAFTAR_JENIS_KLAIM)

    usia_kendaraan = st.slider(
        "Usia Kendaraan (tahun)",
        min_value=FEATURE_LIMITS_NUMERIK["usia_kendaraan_tahun"]["min"],
        max_value=FEATURE_LIMITS_NUMERIK["usia_kendaraan_tahun"]["max"],
        value=FEATURE_LIMITS_NUMERIK["usia_kendaraan_tahun"]["default"],
    )

    jml_kendaraan = st.slider(
        "Jumlah Kendaraan Terlibat",
        min_value=FEATURE_LIMITS_NUMERIK["jumlah_kendaraan_terlibat"]["min"],
        max_value=FEATURE_LIMITS_NUMERIK["jumlah_kendaraan_terlibat"]["max"],
        value=FEATURE_LIMITS_NUMERIK["jumlah_kendaraan_terlibat"]["default"],
    )

    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================================================
# 8. TOMBOL PREDIKSI & HASIL
# ==========================================================================
st.markdown('<div class="jr-card">', unsafe_allow_html=True)
st.markdown("### 🔍 Hasil Prediksi")

tombol_prediksi = st.button("Prediksi Tingkat Keparahan Cedera", use_container_width=True)

if tombol_prediksi:
    input_df = build_input_dataframe(
        usia, usia_kendaraan, jml_kendaraan,
        provinsi, gender, jenis_kecelakaan, jenis_kendaraan, jenis_klaim,
    )

    with st.expander("Lihat data input yang dikirim ke model"):
        st.dataframe(input_df, use_container_width=True)

    if model is None:
        st.error(
            "Model belum tersedia. Letakkan salah satu file berikut pada "
            f"direktori aplikasi ini lalu jalankan ulang: {', '.join(KEMUNGKINAN_NAMA_FILE_MODEL)}."
        )
    else:
        try:
            # Jika ada preprocessor terpisah, transformasikan dahulu.
            # Jika model adalah Pipeline lengkap (ColumnTransformer + estimator),
            # predict() langsung dipanggil pada DataFrame mentah.
            data_final = input_df
            if preprocessor is not None:
                data_final = preprocessor.transform(input_df)

            prediksi = model.predict(data_final)[0]

            # Ambil label kelas langsung dari model (bukan hardcode urutan),
            # lalu petakan ke label yang mudah dibaca lewat LABEL_KELAS.
            kelas_model = list(getattr(model, "classes_", []))
            label_prediksi = LABEL_KELAS.get(prediksi, str(prediksi))

            st.markdown(
                f"""
                <div class="jr-result">
                    <div class="label">Prediksi {NAMA_KOLOM_TARGET.replace('_', ' ').title()}</div>
                    <div class="value">{label_prediksi}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if hasattr(model, "predict_proba") and kelas_model:
                proba = model.predict_proba(data_final)[0]
                proba_df = pd.DataFrame({
                    "Tingkat Keparahan": [LABEL_KELAS.get(k, str(k)) for k in kelas_model],
                    "Probabilitas": np.round(proba, 4),
                }).sort_values("Probabilitas", ascending=False)

                st.markdown("#### Distribusi Probabilitas")
                st.bar_chart(proba_df.set_index("Tingkat Keparahan"))
                st.dataframe(proba_df, use_container_width=True, hide_index=True)

                st.caption(
                    "⚠️ Label kelas di atas (mis. 'Kelas 0', 'Kelas 1') adalah "
                    "placeholder. Sesuaikan kamus `LABEL_KELAS` di bagian atas "
                    "skrip ini dengan arti sebenarnya dari setiap kelas sesuai "
                    "definisi saat training model."
                )

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
        © PT Jasa Raharja (Persero) — Data Management Team
    </div>
    """,
    unsafe_allow_html=True,
)
