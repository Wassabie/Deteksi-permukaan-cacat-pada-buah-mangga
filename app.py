import html
import math

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from utils.predict import detect_mango


# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="MangoVision",
    page_icon="🥭",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================
# MODERN CUSTOM CSS
# ==================================================
st.markdown(
    """
    <style>
        :root {
            --bg: #070B14;
            --panel: #0F1724;
            --panel-soft: #121C2B;
            --border: rgba(148, 163, 184, 0.16);
            --text: #F8FAFC;
            --muted: #94A3B8;
            --yellow: #FACC15;
            --orange: #F97316;
            --green: #22C55E;
            --red: #EF4444;
            --blue: #38BDF8;
            --purple: #A78BFA;
        }

        html, body, [class*="css"] {
            font-family:
                Inter,
                ui-sans-serif,
                system-ui,
                -apple-system,
                BlinkMacSystemFont,
                "Segoe UI",
                sans-serif;
        }

        .stApp {
            background:
                radial-gradient(
                    circle at 15% 10%,
                    rgba(250, 204, 21, 0.08),
                    transparent 27%
                ),
                radial-gradient(
                    circle at 90% 5%,
                    rgba(56, 189, 248, 0.07),
                    transparent 25%
                ),
                var(--bg);
            color: var(--text);
        }

        .block-container {
            max-width: 1500px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }

        [data-testid="stSidebar"] {
            background:
                linear-gradient(
                    180deg,
                    rgba(15, 23, 36, 0.98),
                    rgba(7, 11, 20, 0.98)
                );
            border-right: 1px solid var(--border);
        }

        [data-testid="stSidebar"] .block-container {
            padding-top: 1.5rem;
        }

        /* Hero */
        .hero {
            position: relative;
            overflow: hidden;
            padding: 2rem 2.1rem;
            margin-bottom: 1.7rem;
            border: 1px solid var(--border);
            border-radius: 26px;
            background:
                linear-gradient(
                    135deg,
                    rgba(18, 28, 43, 0.94),
                    rgba(10, 15, 27, 0.96)
                );
            box-shadow:
                0 24px 70px rgba(0, 0, 0, 0.28),
                inset 0 1px 0 rgba(255, 255, 255, 0.025);
        }

        .hero::after {
            content: "";
            position: absolute;
            width: 320px;
            height: 320px;
            top: -210px;
            right: -80px;
            border-radius: 50%;
            background: rgba(250, 204, 21, 0.12);
            filter: blur(4px);
        }

        .hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.42rem 0.8rem;
            margin-bottom: 0.8rem;
            border: 1px solid rgba(250, 204, 21, 0.22);
            border-radius: 999px;
            color: #FDE68A;
            background: rgba(250, 204, 21, 0.08);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }

        .hero-title {
            position: relative;
            z-index: 2;
            margin: 0;
            color: var(--text);
            font-size: clamp(2rem, 4vw, 3.6rem);
            font-weight: 850;
            line-height: 1.05;
            letter-spacing: -0.045em;
        }

        .hero-title span {
            color: var(--yellow);
        }

        .hero-subtitle {
            position: relative;
            z-index: 2;
            max-width: 760px;
            margin-top: 0.85rem;
            color: var(--muted);
            font-size: 1rem;
            line-height: 1.7;
        }

        .hero-chips {
            position: relative;
            z-index: 2;
            display: flex;
            flex-wrap: wrap;
            gap: 0.65rem;
            margin-top: 1.2rem;
        }

        .chip {
            padding: 0.42rem 0.72rem;
            border: 1px solid var(--border);
            border-radius: 999px;
            color: #CBD5E1;
            background: rgba(15, 23, 42, 0.65);
            font-size: 0.77rem;
            font-weight: 650;
        }

        /* Generic panel */
        .panel {
            height: 100%;
            padding: 1rem;
            border: 1px solid var(--border);
            border-radius: 20px;
            background:
                linear-gradient(
                    180deg,
                    rgba(18, 28, 43, 0.94),
                    rgba(10, 15, 27, 0.95)
                );
            box-shadow: 0 14px 38px rgba(0, 0, 0, 0.18);
        }

        .panel-title {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.8rem;
            margin-bottom: 0.85rem;
            color: var(--text);
            font-size: 1rem;
            font-weight: 750;
        }

        .panel-tag {
            padding: 0.3rem 0.55rem;
            border: 1px solid var(--border);
            border-radius: 999px;
            color: var(--muted);
            background: rgba(2, 6, 23, 0.42);
            font-size: 0.7rem;
            font-weight: 650;
        }

        /* Result banner */
        .result-banner {
            display: flex;
            align-items: center;
            gap: 1rem;
            padding: 1.15rem 1.25rem;
            margin: 1.1rem 0 1.25rem;
            border: 1px solid var(--result-border);
            border-radius: 18px;
            background: var(--result-bg);
            box-shadow: inset 4px 0 0 var(--result-color);
        }

        .result-icon {
            display: grid;
            width: 48px;
            height: 48px;
            flex: 0 0 48px;
            place-items: center;
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.06);
            font-size: 1.45rem;
        }

        .result-heading {
            margin: 0;
            color: var(--result-color);
            font-size: 1.05rem;
            font-weight: 800;
        }

        .result-copy {
            margin-top: 0.18rem;
            color: #D7E0EC;
            font-size: 0.9rem;
            line-height: 1.5;
        }

        /* Metric cards */
        .metric-card {
            min-height: 135px;
            padding: 1rem 1.05rem;
            border: 1px solid var(--border);
            border-radius: 18px;
            background:
                linear-gradient(
                    180deg,
                    rgba(18, 28, 43, 0.95),
                    rgba(11, 17, 29, 0.96)
                );
            box-shadow:
                0 12px 30px rgba(0, 0, 0, 0.16),
                inset 0 1px 0 rgba(255, 255, 255, 0.025);
        }

        .metric-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1.2rem;
        }

        .metric-icon {
            display: grid;
            width: 40px;
            height: 40px;
            place-items: center;
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.055);
            font-size: 1.05rem;
        }

        .metric-kicker {
            color: var(--muted);
            font-size: 0.7rem;
            font-weight: 750;
            letter-spacing: 0.07em;
            text-transform: uppercase;
        }

        .metric-value {
            overflow: hidden;
            color: var(--text);
            font-size: clamp(1rem, 2vw, 1.45rem);
            font-weight: 800;
            line-height: 1.25;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .metric-help {
            margin-top: 0.25rem;
            color: var(--muted);
            font-size: 0.76rem;
        }

        /* Confidence block */
        .confidence-panel {
            padding: 1.25rem;
            border: 1px solid var(--border);
            border-radius: 20px;
            background:
                linear-gradient(
                    180deg,
                    rgba(18, 28, 43, 0.94),
                    rgba(10, 15, 27, 0.95)
                );
        }

        .confidence-header {
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 0.8rem;
        }

        .confidence-title {
            color: var(--text);
            font-size: 0.95rem;
            font-weight: 750;
        }

        .confidence-number {
            color: var(--confidence-color);
            font-size: 1.75rem;
            font-weight: 850;
            letter-spacing: -0.035em;
        }

        .progress-track {
            overflow: hidden;
            height: 12px;
            border-radius: 999px;
            background: rgba(148, 163, 184, 0.14);
        }

        .progress-value {
            width: var(--confidence-width);
            height: 100%;
            border-radius: inherit;
            background: var(--confidence-color);
            box-shadow: 0 0 20px var(--confidence-shadow);
            transition: width 0.35s ease;
        }

        .confidence-note {
            margin-top: 0.75rem;
            color: var(--muted);
            font-size: 0.78rem;
            line-height: 1.55;
        }

        /* Empty state */
        .empty-state {
            padding: 4.5rem 1.5rem;
            border: 1px dashed rgba(250, 204, 21, 0.32);
            border-radius: 24px;
            text-align: center;
            background:
                linear-gradient(
                    180deg,
                    rgba(18, 28, 43, 0.58),
                    rgba(10, 15, 27, 0.66)
                );
        }

        .empty-icon {
            margin-bottom: 0.9rem;
            font-size: 3.2rem;
        }

        .empty-title {
            color: var(--text);
            font-size: 1.25rem;
            font-weight: 800;
        }

        .empty-copy {
            max-width: 520px;
            margin: 0.45rem auto 0;
            color: var(--muted);
            line-height: 1.7;
        }

        /* Sidebar card */
        .sidebar-card {
            padding: 1rem;
            margin-bottom: 1rem;
            border: 1px solid var(--border);
            border-radius: 16px;
            background: rgba(18, 28, 43, 0.82);
        }

        .sidebar-card-title {
            color: var(--yellow);
            font-size: 0.9rem;
            font-weight: 800;
        }

        .sidebar-card-copy {
            margin-top: 0.35rem;
            color: var(--muted);
            font-size: 0.78rem;
            line-height: 1.55;
        }

        .model-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.7rem;
            padding: 0.72rem 0;
            border-bottom: 1px solid rgba(148, 163, 184, 0.1);
        }

        .model-row:last-child {
            border-bottom: none;
        }

        .model-key {
            color: var(--muted);
            font-size: 0.72rem;
        }

        .model-value {
            color: var(--text);
            font-size: 0.76rem;
            font-weight: 750;
            text-align: right;
        }

        /* Streamlit uploader */
        [data-testid="stFileUploader"] {
            padding: 0.35rem;
            border: 1px dashed rgba(250, 204, 21, 0.34);
            border-radius: 16px;
            background: rgba(15, 23, 42, 0.55);
        }

        [data-testid="stFileUploader"]:hover {
            border-color: rgba(250, 204, 21, 0.7);
            background: rgba(30, 41, 59, 0.58);
        }

        [data-testid="stFileUploaderDropzone"] {
            border: none;
            background: transparent;
        }

        [data-testid="stFileUploaderDropzoneInstructions"] {
            color: var(--text);
        }

        /* Tabs */
        button[data-baseweb="tab"] {
            color: var(--muted);
            font-weight: 700;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            color: var(--yellow);
        }

        [data-testid="stDataFrame"] {
            border: 1px solid var(--border);
            border-radius: 16px;
            overflow: hidden;
        }

        hr {
            border-color: rgba(148, 163, 184, 0.12) !important;
        }

        @media (max-width: 900px) {
            .hero {
                padding: 1.55rem;
                border-radius: 20px;
            }

            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .metric-card {
                min-height: auto;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# HELPER FUNCTIONS
# ==================================================
def get_result_theme(status: object, detected_class: object) -> dict[str, str]:
    """Menentukan tema berdasarkan status dan kelas hasil prediksi."""

    normalized_status = str(status or "").strip().lower().replace("_", " ")
    normalized_class = str(detected_class or "").strip().lower().replace("_", " ")

    not_mango_terms = (
        "not mango",
        "non mango",
        "bukan mangga",
        "tidak mangga",
        "unknown",
        "tidak terdeteksi",
    )
    healthy_terms = ("healthy", "sehat")
    defect_terms = (
        "defect",
        "cacat",
        "anthracnose",
        "alternaria",
        "stem and rot",
        "stem end rot",
        "black mould rot",
        "black mold rot",
    )

    # Status negatif diperiksa lebih dahulu agar teks seperti
    # "not healthy" tidak keliru dianggap sebagai mangga sehat.
    if any(term in normalized_status for term in not_mango_terms):
        result_key = "not_mango"
    elif any(term in normalized_status for term in healthy_terms):
        result_key = "healthy"
    elif any(term in normalized_status for term in defect_terms):
        result_key = "defect"
    elif any(term in normalized_class for term in healthy_terms):
        result_key = "healthy"
    elif any(term in normalized_class for term in defect_terms):
        result_key = "defect"
    else:
        result_key = "not_mango"

    themes = {
        "healthy": {
            "icon": "✓",
            "label": "Healthy Mango",
            "color": "#22C55E",
            "border": "rgba(34, 197, 94, 0.32)",
            "background": "rgba(34, 197, 94, 0.08)",
            "message": (
                "Model mengenali buah mangga dalam kondisi sehat. "
                "Tidak ditemukan kelas cacat dengan confidence yang "
                "melewati ambang deteksi."
            ),
        },
        "defect": {
            "icon": "!",
            "label": "Defect Detected",
            "color": "#F97316",
            "border": "rgba(249, 115, 22, 0.34)",
            "background": "rgba(249, 115, 22, 0.08)",
            "message": (
                "Model menemukan indikasi cacat permukaan pada buah "
                "mangga. Periksa bounding box untuk melihat area yang "
                "terdeteksi."
            ),
        },
        "not_mango": {
            "icon": "×",
            "label": "Not Mango",
            "color": "#EF4444",
            "border": "rgba(239, 68, 68, 0.34)",
            "background": "rgba(239, 68, 68, 0.08)",
            "message": (
                "Gambar tidak dikenali sebagai buah mangga atau nilai "
                "confidence berada di bawah ambang penerimaan."
            ),
        },
    }

    return themes[result_key]


def normalize_confidence(value: object) -> float:
    """Mengubah confidence menjadi nilai aman pada rentang 0 sampai 1."""

    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0

    if not math.isfinite(confidence):
        return 0.0

    # Mendukung fungsi prediksi yang mengembalikan 85.5 maupun 0.855.
    if 1.0 < confidence <= 100.0:
        confidence /= 100.0

    return max(0.0, min(confidence, 1.0))


def render_image_panel(title: str, tag: str, image_data) -> None:
    """Menampilkan gambar di dalam panel bergaya modern."""

    st.markdown(
        f"""
        <div class="panel-title">
            <span>{html.escape(title)}</span>
            <span class="panel-tag">{html.escape(tag)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.image(image_data, use_container_width=True)


def render_metric_card(
    icon: str,
    label: str,
    value: str,
    help_text: str,
    accent: str,
) -> None:
    """Menampilkan kartu metrik."""

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-top">
                <div class="metric-icon">{html.escape(icon)}</div>
                <div class="metric-kicker">{html.escape(label)}</div>
            </div>
            <div class="metric-value" style="color:{accent};">
                {html.escape(value)}
            </div>
            <div class="metric-help">
                {html.escape(help_text)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==================================================
# HEADER
# ==================================================
st.markdown(
    """
    <section class="hero">
        <div class="hero-badge">Computer Vision Quality Control</div>
        <h1 class="hero-title">
            Mango<span>Vision
        </h1>
        <div class="hero-subtitle">
            Sistem berbasis YOLOv11 untuk mengidentifikasi kondisi
            dan cacat permukaan buah mangga melalui citra digital.
        </div>
        <div class="hero-chips">
            <span class="chip">⌖ Defect Detection</span>
            <span class="chip">⚡ Confidence Analysis</span>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# SIDEBAR
# ==================================================
with st.sidebar:
    st.markdown("## Control Center")

    st.markdown(
        """
        <div class="sidebar-card">
            <div class="sidebar-card-title">
                Unggah gambar pengujian
            </div>
            <div class="sidebar-card-copy">
                Gunakan foto JPG, JPEG, atau PNG dengan objek terlihat
                jelas dan pencahayaan yang cukup.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Pilih gambar",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    st.info(
        "Gunakan gambar yang tidak terlalu gelap, tidak buram, "
        "dan menampilkan buah secara dominan."
    )


# ==================================================
# DEFAULT VIEW
# ==================================================
if uploaded is None:
    empty_state_html = (
        '<div class="empty-state">'
        '<div class="empty-icon">📷</div>'
        '<div class="empty-title">Belum ada gambar untuk dianalisis</div>'
        '<div class="empty-copy">'
        'Unggah gambar melalui panel sebelah kiri. Sistem akan memeriksa '
        'apakah gambar merupakan buah mangga, lalu menjalankan deteksi '
        'kondisi permukaannya.'
        '</div>'
        '</div>'
    )

    st.markdown(
        empty_state_html,
        unsafe_allow_html=True,
    )
    st.stop()


# ==================================================
# IMAGE PROCESSING
# ==================================================
try:
    uploaded.seek(0)
    image = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
except Exception as error:
    st.error(f"Gambar tidak dapat dibuka: {error}")
    st.stop()

with st.spinner("AI sedang memindai bentuk dan permukaan buah..."):
    try:
        prediction = detect_mango(image)

        if not isinstance(prediction, (tuple, list)) or len(prediction) != 4:
            raise ValueError(
                "detect_mango() harus mengembalikan empat nilai: "
                "result_img, status, conf, dan detected_class."
            )

        result_img, status, conf, detected_class = prediction
    except Exception as error:
        st.error(f"Proses deteksi gagal: {error}")
        st.stop()

if result_img is None:
    result_img = image

status_text = str(status or "Tidak diketahui")
class_text = str(detected_class or "Tidak terdeteksi")

# Batasi confidence agar aman untuk tampilan progress.
confidence = normalize_confidence(conf)
confidence_percent = confidence * 100

theme = get_result_theme(status_text, class_text)

safe_theme_label = html.escape(theme["label"])
safe_theme_message = html.escape(theme["message"])


# ==================================================
# IMAGE COMPARISON
# ==================================================
st.markdown("## Hasil Pemindaian")

col_original, col_result = st.columns(2, gap="large")

with col_original:
    with st.container(border=True):
        render_image_panel(
            title="Gambar Asli",
            tag="INPUT",
            image_data=image,
        )

with col_result:
    with st.container(border=True):
        render_image_panel(
            title="Hasil Deteksi",
            tag="OUTPUT",
            image_data=result_img,
        )


# ==================================================
# RESULT BANNER
# ==================================================
st.markdown(
    f"""
    <div
        class="result-banner"
        style="
            --result-color:{theme['color']};
            --result-border:{theme['border']};
            --result-bg:{theme['background']};
        "
    >
        <div class="result-icon">{theme['icon']}</div>
        <div>
            <div class="result-heading">{safe_theme_label}</div>
            <div class="result-copy">{safe_theme_message}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# METRIC CARDS
# ==================================================
metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(
    4,
    gap="medium",
)

with metric_col_1:
    render_metric_card(
        icon=theme["icon"],
        label="Status",
        value=theme["label"],
        help_text="Hasil akhir sistem",
        accent=theme["color"],
    )

with metric_col_2:
    render_metric_card(
        icon="⌁",
        label="Kelas",
        value=class_text,
        help_text="Kelas dengan confidence tertinggi",
        accent="#FACC15",
    )

with metric_col_3:
    render_metric_card(
        icon="%",
        label="Confidence",
        value=f"{confidence_percent:.1f}%",
        help_text="Keyakinan pada satu prediksi",
        accent="#38BDF8",
    )

with metric_col_4:
    render_metric_card(
        icon="AI",
        label="Model",
        value="YOLOv11",
        help_text="Model deteksi yang digunakan",
        accent="#A78BFA",
    )


# ==================================================
# DETAILED ANALYSIS
# ==================================================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("## Detail Analisis")

tab_summary, tab_data = st.tabs(
    ["Confidence Overview", "Data Prediksi"]
)

with tab_summary:
    confidence_html = (
        '<div class="confidence-panel" '
        f'style="--confidence-color:{theme["color"]};'
        f'--confidence-shadow:{theme["color"]}55;'
        f'--confidence-width:{confidence_percent:.2f}%;">'
        '<div class="confidence-header">'
        '<div class="confidence-title">Confidence hasil prediksi</div>'
        f'<div class="confidence-number">{confidence_percent:.1f}%</div>'
        '</div>'
        '<div class="progress-track">'
        '<div class="progress-value"></div>'
        '</div>'
        '<div class="confidence-note">'
        'Confidence menunjukkan tingkat keyakinan model pada gambar ini, '
        'bukan akurasi keseluruhan model terhadap seluruh dataset pengujian.'
        '</div>'
        '</div>'
    )

    st.markdown(
        confidence_html,
        unsafe_allow_html=True,
    )

    st.caption(
        "Status mentah dari fungsi prediksi: "
        f"{status_text}"
    )

with tab_data:
    result_data = pd.DataFrame(
        {
            "Parameter": [
                "Status sistem",
                "Kelas hasil",
                "Confidence",
                "Model utama",
            ],
            "Nilai": [
                status_text,
                class_text,
                f"{confidence_percent:.2f}%",
                "YOLOv11",
            ],
        }
    )

    st.dataframe(
        result_data,
        use_container_width=True,
        hide_index=True,
    )
