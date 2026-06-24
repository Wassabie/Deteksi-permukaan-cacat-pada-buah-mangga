import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

from utils.predict import detect_mango

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Mango Defect Detection",
    page_icon="🥭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# MODERN CUSTOM CSS
# ==================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    /* Global Typography & Background */
    .stApp {
        background-color: #0B0F19;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Header Styles */
    .header-container {
        text-align: center;
        padding: 2rem 0 3rem 0;
    }
    .main-title {
        font-size: 46px;
        font-weight: 800;
        background: linear-gradient(135deg, #FACC15 0%, #F59E0B 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        color: #94A3B8;
        font-size: 18px;
        font-weight: 400;
    }
    
    /* Custom Status Cards */
    .custom-card {
        background: #111827;
        border: 1px solid #1F2937;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: transform 0.2s, border-color 0.2s;
    }
    .custom-card:hover {
        border-color: #374151;
        transform: translateY(-2px);
    }
    .card-val {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .card-lbl {
        color: #94A3B8;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    .card-desc {
        color: #F3F4F6;
        font-size: 16px;
        font-weight: 600;
        margin-top: 0.5rem;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0E131F;
        border-right: 1px solid #1F2937;
    }
    
    /* Streamlit Native Component Overrides for Dark Theme */
    div[data-testid="stContainer"] {
        border-radius: 16px !important;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #FACC15 0%, #EAB308 100%) !important;
        color: #020617 !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        border: none !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.2s ease;
    }
    .stButton button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 15px rgba(234, 179, 8, 0.3);
    }
    
    /* Headings */
    h2, h3 {
        font-weight: 700 !important;
        color: #F8FAFC !important;
    }
    
    /* Drag & Drop Upload Area */
            
    [data-testid="stFileUploader"] {
        background: #111827;
        border: 2px dashed #FACC15;
        border-radius: 16px;
        padding: 20px;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #EAB308;
        background: #1F2937;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: transparent;
        border: none;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] {
        color: #F8FAFC;
    }
</style>
""", unsafe_allow_html=True)

# ==================================================
# HEADER
# ==================================================
st.markdown("""
<div class="header-container">
    <div class="main-title">🥭 Mango Defect Detection</div>
    <div class="subtitle">Sistem Deteksi Cacat Permukaan Mangga Berbasis Deep Learning (YOLOv11)</div>
</div>
""", unsafe_allow_html=True)

# ==================================================
# SIDEBAR
# ==================================================
with st.sidebar:
    st.markdown("### ⚙️ Control Panel")
    st.markdown("""
    <div style="
        padding:15px;
        border-radius:12px;
        background:#111827;
        border:1px solid #374151;
        margin-bottom:10px;
    ">
        <h4 style="margin:0;color:#FACC15;">
            📤 Upload / Drag & Drop Gambar
        </h4>
        <p style="margin:5px 0 0 0;color:#94A3B8;">
            Seret gambar mangga ke area di bawah atau klik Browse files.
        </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### 📌 Informasi Model")
    
    # Menggunakan UI asli streamlit yang rapi untuk sidebar
    st.caption("MODEL ARCHITECTURE")
    st.text("YOLOv11 (Ultralytics)")
    
    st.caption("CORE TASK")
    st.text("Object Detection & Quality Control")
    
    st.caption("TARGET CLASSES")
    st.markdown("- `Healthy` (Kondisi Baik)\n- `Defect` (Cacat Permukaan)")
    
    st.markdown("---")
    st.info("💡 **Petunjuk:** Pastikan pencahayaan gambar cukup terang untuk hasil deteksi yang optimal.")

# ==================================================
# DEFAULT VIEW (IF NO IMAGE)
# ==================================================
if uploaded is None:
    st.markdown("### 📥 Mulai Analisis")
    with st.container(border=True):
        st.markdown("""
        <div style="text-align: center; padding: 3rem 1rem;">
            <p style="font-size: 48px; margin-bottom: 1rem;">📸</p>
            <h4 style="color: #F8FAFC; margin-bottom: 0.5rem;">Belum ada gambar yang diunggah</h4>
            <p style="color: #64748B; max-width: 400px; margin: 0 auto;">
                Silakan pilih atau seret gambar buah mangga ke area dropzone di panel sebelah kiri untuk memulai pemindaian AI.
            </p>
        </div>
        """, unsafe_allow_html=True)
    st.stop()

# ==================================================
# IMAGE PROCESSING
# ==================================================
image = Image.open(uploaded)

with st.spinner("🔍 AI sedang menganalisis tekstur permukaan gambar..."):
    # Memanggil fungsi deteksi Anda
    result_img, status, conf, detected_class = detect_mango(image)

# ==================================================
# IMAGE DISPLAY (SIDE-BY-SIDE)
# ==================================================
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("### 📤 Gambar Asli")
    with st.container(border=True):
        st.image(image, use_container_width=True)

with col2:
    st.markdown("### 🤖 Hasil Deteksi AI")
    with st.container(border=True):
        st.image(result_img, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==================================================
# STATUS RESULT & METRICS
# ==================================================
st.markdown("## 🧠 Ringkasan Analisis")

# Menentukan warna & ikon berdasarkan status
if "Healthy" in status:
    status_icon = "🟢"
    status_color = "#10B981"  # Emerald green
    result_label = "Healthy"
    alert_type = st.success
    alert_msg = f"**Aman!** Buah mangga terdeteksi dalam kondisi sehat dan layak distribusi. ({status})"
elif "Defect" in status:
    status_icon = "🟠"
    status_color = "#F97316"  # Orange
    result_label = "Defect Detected"
    alert_type = st.warning
    alert_msg = f"**Perhatian!** Ditemukan cacat permukaan pada buah mangga dengan jenis: **{detected_class}**. ({status})"
else:
    status_icon = "🔴"
    status_color = "#EF4444"  # Red
    result_label = "Not Detected"
    alert_type = st.error
    alert_msg = f"**Gagal!** Objek mangga tidak dikenali atau di luar jangkauan deteksi model. ({status})"

col_status, col_class, col_conf, col_task = st.columns(4, gap="medium")

with col_status:
    st.markdown(f"""
    <div class="custom-card">
        <div class="card-val" style="color: {status_color};">{status_icon}</div>
        <div class="card-lbl">Status Deteksi</div>
        <div class="card-desc" style="color: {status_color};">{result_label}</div>
    </div>
    """, unsafe_allow_html=True)

with col_class:
    st.markdown(f"""
    <div class="custom-card">
        <div class="card-val" style="color: #FACC15;">🥭</div>
        <div class="card-lbl">Jenis Deteksi</div>
        <div class="card-desc" style="color: #FACC15;">{detected_class}</div>
    </div>
    """, unsafe_allow_html=True)

with col_conf:
    st.markdown(f"""
    <div class="custom-card">
        <div class="card-val" style="color: #38BDF8;">{conf * 100:.1f}%</div>
        <div class="card-lbl">Confidence Score</div>
        <div class="card-desc" style="color: #38BDF8;">Tingkat Keyakinan</div>
    </div>
    """, unsafe_allow_html=True)

with col_task:
    st.markdown("""
    <div class="custom-card">
        <div class="card-val" style="color: #A78BFA;">YOLOv11</div>
        <div class="card-lbl">Model Arsitektur</div>
        <div class="card-desc" style="color: #A78BFA;">Computer Vision</div>
    </div>
    """, unsafe_allow_html=True)

# Alert Status Bar
st.markdown("<br>", unsafe_allow_html=True)
alert_type(alert_msg)

# ==================================================
# DETAILED INTERACTION (TABS METHOD)
# ==================================================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("## 📊 Detail Data & Visualisasi")

tab1, tab2 = st.tabs(["📈 Grafik Confidence", "📋 Tabulasi Data"])

with tab1:
    # Desain grafik yang jauh lebih bersih dan modern (Borderless & Flat)
    fig, ax = plt.subplots(figsize=(10, 2))
    fig.patch.set_facecolor("#0B0F19")
    ax.set_facecolor("#0B0F19")

    bars = ax.barh([status], [conf * 100], color=status_color, height=0.35)
    
    # Label value di dalam/ujung bar
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2f}%",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="#F8FAFC"
        )

    ax.set_xlim(0, 100)
    
    # Hilangkan seluruh border/spines box agar menyatu dengan background
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(colors="#94A3B8", labelsize=11, bottom=False, left=False)
    ax.xaxis.grid(True, linestyle=":", alpha=0.15, color="#94A3B8")
    ax.set_axisbelow(True)

    st.pyplot(fig)
    
    # Progress Bar bawaan streamlit yang minimalis
    st.progress(float(conf))
    st.caption("ℹ️ *Akurasi prediksi berbanding lurus dengan kualitas kedekatan objek dan pencahayaan studio.*")

with tab2:
    result_data = pd.DataFrame({
    "Parameter": [
        "Status Prediksi",
        "Jenis Deteksi",
        "Confidence Score",
        "Engine Model",
        "Tipe Tugas AI"
    ],
    "Value": [
        status,
        detected_class,
        f"{conf * 100:.2f}%",
        "YOLOv11 Framework",
        "Object Detection"
    ]
})
    
    st.dataframe(
        result_data,
        use_container_width=True,
        hide_index=True
    )