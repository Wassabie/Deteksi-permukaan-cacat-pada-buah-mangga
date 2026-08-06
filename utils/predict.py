import cv2
import numpy as np

from utils.model_service import load_models


# ==================================================
# MEMUAT MODEL
# ==================================================

input_validator, defect_detector = load_models()


# ==================================================
# KONFIGURASI
# ==================================================

# Threshold hanya digunakan untuk model deteksi cacat
DEFECT_CONF_THRESHOLD = 0.25

# Ukuran input model klasifikasi
CLASSIFICATION_IMAGE_SIZE = 224

# Ukuran input model deteksi
DETECTION_IMAGE_SIZE = 640


MANGO_CLASS_NAMES = {
    "mango",
    "buah_mangga",
    "mango_fruit",
}


HEALTHY_CLASS_NAMES = {
    "healthy",
    "healthy_mango",
    "mango_healthy",
    "sehat",
}


# ==================================================
# NORMALISASI NAMA KELAS
# ==================================================

def normalize_class_name(class_name):
    """
    Menyamakan format nama kelas.

    Contoh:
    Mango Fruit -> mango_fruit
    mango-fruit -> mango_fruit
    """

    return (
        str(class_name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


# ==================================================
# MENGAMBIL NAMA KELAS
# ==================================================

def get_class_name(names, class_id):
    """
    Mengambil nama kelas dari dictionary atau list.
    """

    if isinstance(names, dict):
        return str(names[class_id])

    return str(names[int(class_id)])


# ==================================================
# MENCARI ID KELAS
# ==================================================

def find_class_id(names, target_names):
    """
    Mencari ID kelas berdasarkan beberapa
    kemungkinan nama kelas.
    """

    if isinstance(names, dict):
        class_items = names.items()
    else:
        class_items = enumerate(names)

    normalized_targets = {
        normalize_class_name(name)
        for name in target_names
    }

    for class_id, class_name in class_items:
        normalized_name = normalize_class_name(
            class_name
        )

        if normalized_name in normalized_targets:
            return int(class_id)

    return None


# ==================================================
# SKALA TAMPILAN DINAMIS
# ==================================================

def get_display_scale(img):
    """
    Menghasilkan skala berdasarkan resolusi gambar.

    Digunakan agar bounding box, banner, dan tulisan
    tetap terlihat jelas pada gambar beresolusi besar.
    """

    height, width = img.shape[:2]
    longest_side = max(height, width)

    return max(
        0.75,
        longest_side / 1000.0,
    )


# ==================================================
# STATUS BOX RINGKAS
# ==================================================

def draw_compact_status(
    img,
    text,
    status_type,
    conf=None,
):
    """
    Menggambar banner status pada kiri bawah gambar.

    Confidence hanya ditampilkan apabila nilai conf
    tidak bernilai None.
    """

    height, width = img.shape[:2]
    scale = get_display_scale(img)

    if status_type == "not_mango":
        bg_color = (0, 0, 180)
        border_color = (0, 0, 255)
        icon = "X"

    elif status_type == "healthy":
        bg_color = (0, 125, 0)
        border_color = (0, 200, 0)
        icon = "OK"

    else:
        bg_color = (0, 90, 180)
        border_color = (0, 140, 255)
        icon = "!"

    line_1 = f"{icon} {text}"

    # Confidence tidak dibuat apabila conf bernilai None
    line_2 = None

    if conf is not None:
        line_2 = f"Confidence: {conf:.2f}"

    font = cv2.FONT_HERSHEY_SIMPLEX

    font_scale_title = max(
        0.60,
        min(
            1.25,
            scale * 0.78,
        ),
    )

    font_scale_conf = max(
        0.52,
        min(
            1.10,
            scale * 0.67,
        ),
    )

    text_thickness = max(
        1,
        min(
            4,
            int(round(scale * 2)),
        ),
    )

    border_thickness = max(
        2,
        min(
            6,
            int(round(scale * 3)),
        ),
    )

    padding_x = max(
        12,
        int(round(scale * 15)),
    )

    padding_y = max(
        10,
        int(round(scale * 11)),
    )

    gap = max(
        6,
        int(round(scale * 7)),
    )

    margin = max(
        12,
        int(round(scale * 14)),
    )

    shadow_offset = max(
        3,
        int(round(scale * 4)),
    )

    (
        title_width,
        title_height,
    ), title_baseline = cv2.getTextSize(
        line_1,
        font,
        font_scale_title,
        text_thickness,
    )

    confidence_width = 0
    confidence_height = 0
    confidence_baseline = 0

    if line_2 is not None:
        (
            confidence_width,
            confidence_height,
        ), confidence_baseline = cv2.getTextSize(
            line_2,
            font,
            font_scale_conf,
            text_thickness,
        )

    box_width = (
        max(
            title_width,
            confidence_width,
        )
        + (padding_x * 2)
    )

    box_height = (
        title_height
        + title_baseline
        + (padding_y * 2)
    )

    if line_2 is not None:
        box_height += (
            gap
            + confidence_height
            + confidence_baseline
        )

    max_box_width = max(
        120,
        width - (margin * 2),
    )

    box_width = min(
        box_width,
        max_box_width,
    )

    x1 = margin

    y1 = max(
        margin,
        height - box_height - margin,
    )

    x2 = min(
        width - margin,
        x1 + box_width,
    )

    y2 = min(
        height - margin,
        y1 + box_height,
    )

    # Membuat bayangan banner
    overlay = img.copy()

    cv2.rectangle(
        overlay,
        (
            x1 + shadow_offset,
            y1 + shadow_offset,
        ),
        (
            min(
                width - 1,
                x2 + shadow_offset,
            ),
            min(
                height - 1,
                y2 + shadow_offset,
            ),
        ),
        (0, 0, 0),
        -1,
        cv2.LINE_AA,
    )

    cv2.addWeighted(
        overlay,
        0.30,
        img,
        0.70,
        0,
        img,
    )

    # Latar utama banner
    cv2.rectangle(
        img,
        (x1, y1),
        (x2, y2),
        bg_color,
        -1,
        cv2.LINE_AA,
    )

    # Garis tepi banner
    cv2.rectangle(
        img,
        (x1, y1),
        (x2, y2),
        border_color,
        border_thickness,
        cv2.LINE_AA,
    )

    title_x = x1 + padding_x

    title_y = (
        y1
        + padding_y
        + title_height
    )

    cv2.putText(
        img,
        line_1,
        (title_x, title_y),
        font,
        font_scale_title,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )

    # Confidence hanya digambar jika tersedia
    if line_2 is not None:
        confidence_y = (
            title_y
            + title_baseline
            + gap
            + confidence_height
        )

        cv2.putText(
            img,
            line_2,
            (
                x1 + padding_x,
                confidence_y,
            ),
            font,
            font_scale_conf,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )

    return img


# ==================================================
# MENGGAMBAR BOUNDING BOX
# ==================================================

def draw_detection(
    img,
    box,
    names,
    color=(0, 140, 255),
):
    """
    Menggambar bounding box dan label.

    Ukuran garis, tulisan, dan banner menyesuaikan
    resolusi gambar secara otomatis.
    """

    height, width = img.shape[:2]

    coordinates = (
        box.xyxy[0]
        .detach()
        .cpu()
        .tolist()
    )

    x1, y1, x2, y2 = map(
        int,
        coordinates,
    )

    # Memastikan koordinat tetap berada di dalam gambar
    x1 = max(
        0,
        min(x1, width - 1),
    )

    y1 = max(
        0,
        min(y1, height - 1),
    )

    x2 = max(
        0,
        min(x2, width - 1),
    )

    y2 = max(
        0,
        min(y2, height - 1),
    )

    class_id = int(
        box.cls[0].item()
    )

    confidence = float(
        box.conf[0].item()
    )

    class_name = get_class_name(
        names,
        class_id,
    )

    display_class_name = class_name.replace(
        "_",
        " ",
    )

    label_text = (
        f"{display_class_name} | "
        f"Conf: {confidence:.2f}"
    )

    scale = get_display_scale(img)

    # Ketebalan bounding box dinamis
    box_thickness = max(
        4,
        min(
            10,
            int(round(scale * 5)),
        ),
    )

    # Ukuran tulisan dinamis
    font_scale = max(
        0.70,
        min(
            1.35,
            scale * 0.90,
        ),
    )

    # Ketebalan tulisan dinamis
    text_thickness = max(
        2,
        min(
            4,
            int(round(scale * 2)),
        ),
    )

    padding_x = max(
        9,
        int(round(scale * 12)),
    )

    padding_y = max(
        7,
        int(round(scale * 9)),
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Menggambar bounding box
    cv2.rectangle(
        img,
        (x1, y1),
        (x2, y2),
        color,
        box_thickness,
        cv2.LINE_AA,
    )

    (
        text_width,
        text_height,
    ), baseline = cv2.getTextSize(
        label_text,
        font,
        font_scale,
        text_thickness,
    )

    label_width = (
        text_width
        + (padding_x * 2)
    )

    label_height = (
        text_height
        + baseline
        + (padding_y * 2)
    )

    label_x1 = x1

    # Mencegah label keluar dari sisi kanan
    if label_x1 + label_width > width:
        label_x1 = max(
            0,
            width - label_width,
        )

    label_x2 = min(
        width - 1,
        label_x1 + label_width,
    )

    # Label ditempatkan di atas bounding box
    if y1 - label_height >= 0:
        label_y1 = y1 - label_height
        label_y2 = y1

    # Jika tidak cukup ruang, label masuk ke dalam box
    else:
        label_y1 = y1

        label_y2 = min(
            height - 1,
            y1 + label_height,
        )

    # Latar label
    cv2.rectangle(
        img,
        (label_x1, label_y1),
        (label_x2, label_y2),
        color,
        -1,
        cv2.LINE_AA,
    )

    text_x = label_x1 + padding_x

    text_y = (
        label_y1
        + padding_y
        + text_height
    )

    cv2.putText(
        img,
        label_text,
        (text_x, text_y),
        font,
        font_scale,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )

    return img


# ==================================================
# MENGAMBIL DETEKSI TERBAIK
# ==================================================

def get_best_detection(boxes, names):
    """
    Mengambil satu deteksi dengan confidence tertinggi.
    """

    if boxes is None or len(boxes) == 0:
        return (
            None,
            None,
            0.0,
        )

    confidence_values = (
        boxes.conf
        .detach()
        .cpu()
        .numpy()
    )

    best_index = int(
        np.argmax(confidence_values)
    )

    best_box = boxes[best_index]

    class_id = int(
        best_box.cls[0].item()
    )

    confidence = float(
        best_box.conf[0].item()
    )

    detected_class = get_class_name(
        names,
        class_id,
    )

    return (
        best_box,
        detected_class,
        confidence,
    )


# ==================================================
# MENYIAPKAN GAMBAR
# ==================================================

def prepare_image(image):
    """
    Mengubah gambar PIL atau NumPy menjadi RGB
    dengan tiga channel.
    """

    img = np.array(image)

    # Grayscale menjadi RGB
    if img.ndim == 2:
        img = cv2.cvtColor(
            img,
            cv2.COLOR_GRAY2RGB,
        )

    # RGBA menjadi RGB
    elif (
        img.ndim == 3
        and img.shape[2] == 4
    ):
        img = cv2.cvtColor(
            img,
            cv2.COLOR_RGBA2RGB,
        )

    if (
        img.ndim != 3
        or img.shape[2] != 3
    ):
        raise ValueError(
            "Format gambar tidak didukung. "
            "Gunakan JPG, JPEG, atau PNG."
        )

    return img


# ==================================================
# VALIDASI MANGO / NON-MANGO
# ==================================================

def validate_mango(img_bgr):
    """
    Memvalidasi gambar menggunakan model klasifikasi.

    Tidak menggunakan threshold confidence.
    Keputusan hanya berdasarkan kelas top-1.
    """

    classification_results = (
        input_validator.predict(
            source=img_bgr,
            imgsz=CLASSIFICATION_IMAGE_SIZE,
            verbose=False,
        )
    )

    if not classification_results:
        raise RuntimeError(
            "Model klasifikasi tidak menghasilkan prediksi."
        )

    cls_result = classification_results[0]

    if cls_result.probs is None:
        raise RuntimeError(
            "Model klasifikasi tidak menghasilkan probabilitas."
        )

    class_names = cls_result.names

    mango_class_id = find_class_id(
        class_names,
        MANGO_CLASS_NAMES,
    )

    if mango_class_id is None:
        raise ValueError(
            "Kelas mango tidak ditemukan pada model klasifikasi. "
            "Pastikan model memiliki kelas mango dan non_mango."
        )

    top1_class_id = int(
        cls_result.probs.top1
    )

    top1_class_name = get_class_name(
        class_names,
        top1_class_id,
    )

    normalized_top1_name = normalize_class_name(
        top1_class_name
    )

    normalized_mango_names = {
        normalize_class_name(name)
        for name in MANGO_CLASS_NAMES
    }

    # Tidak menggunakan threshold confidence
    is_mango = (
        normalized_top1_name
        in normalized_mango_names
    )

    print(
        "[VALIDATOR] "
        f"Hasil: {normalized_top1_name} | "
        f"Diterima sebagai mangga: {is_mango}"
    )

    return (
        is_mango,
        normalized_top1_name,
    )


# ==================================================
# FUNGSI UTAMA IDENTIFIKASI
# ==================================================

def detect_mango(
    image,
    defect_conf_threshold=DEFECT_CONF_THRESHOLD,
):
    """
    Alur proses:

    1. Gambar divalidasi oleh model klasifikasi.
    2. Tidak menggunakan threshold klasifikasi mangga.
    3. Jika top-1 adalah mango, gambar diteruskan.
    4. Jika bukan mango, proses deteksi dihentikan.
    5. Model deteksi menampilkan satu confidence tertinggi.
    """

    img_rgb = prepare_image(image)

    img_bgr = cv2.cvtColor(
        img_rgb,
        cv2.COLOR_RGB2BGR,
    )

    output_img = img_bgr.copy()

    # ==================================================
    # TAHAP 1: VALIDASI MANGO
    # ==================================================

    (
        is_mango,
        validator_class,
    ) = validate_mango(img_bgr)

    # ==================================================
    # BUKAN MANGGA
    # ==================================================

    if not is_mango:
        # Banner Not Mango tanpa confidence
        output_img = draw_compact_status(
            output_img,
            "Not Mango",
            "not_mango",
            conf=None,
        )

        status = "❌ Not Mango"
        detected_class = "Not Mango"

        # None agar confidence tidak ditampilkan di app.py
        final_confidence = None

    # ==================================================
    # MANGGA
    # ==================================================

    else:
        detection_results = (
            defect_detector.predict(
                source=img_bgr,
                imgsz=DETECTION_IMAGE_SIZE,
                conf=defect_conf_threshold,
                verbose=False,
            )
        )

        if not detection_results:
            raise RuntimeError(
                "Model deteksi tidak menghasilkan prediksi."
            )

        det_result = detection_results[0]

        (
            best_box,
            detected_class,
            defect_confidence,
        ) = get_best_detection(
            det_result.boxes,
            det_result.names,
        )

        # ==================================================
        # TIDAK ADA HASIL DETEKSI
        # ==================================================

        if best_box is None:
            output_img = draw_compact_status(
                output_img,
                "No Detection",
                "no_detection",
                conf=None,
            )

            status = "⚠️ Tidak ada kondisi yang terdeteksi"
            detected_class = "No Detection"
            final_confidence = None

        # ==================================================
        # ADA HASIL DETEKSI
        # ==================================================

        else:
            normalized_detected_class = (
                normalize_class_name(
                    detected_class
                )
            )

            normalized_healthy_names = {
                normalize_class_name(name)
                for name in HEALTHY_CLASS_NAMES
            }

            # ==============================================
            # HEALTHY
            # ==============================================

            if (
                normalized_detected_class
                in normalized_healthy_names
            ):
                output_img = draw_detection(
                    output_img,
                    best_box,
                    det_result.names,
                    color=(0, 200, 0),
                )

                status = (
                    f"✅ Healthy Mango "
                    f"({defect_confidence * 100:.1f}%)"
                )

                detected_class = "Healthy"
                final_confidence = defect_confidence

            # ==============================================
            # CACAT
            # ==============================================

            else:
                output_img = draw_detection(
                    output_img,
                    best_box,
                    det_result.names,
                    color=(0, 140, 255),
                )

                status = (
                    f"⚠️ Defect Mango: "
                    f"{detected_class} "
                    f"({defect_confidence * 100:.1f}%)"
                )

                final_confidence = defect_confidence

    # BGR kembali menjadi RGB untuk Streamlit
    output_img = cv2.cvtColor(
        output_img,
        cv2.COLOR_BGR2RGB,
    )

    return (
        output_img,
        status,
        final_confidence,
        detected_class,
    )