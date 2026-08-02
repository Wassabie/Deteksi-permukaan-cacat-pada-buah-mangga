import cv2
import numpy as np
from utils.model_service import load_models


input_validator, defect_detector = load_models()
# ==================================================
# KONFIGURASI
# ==================================================

MANGO_CONF_THRESHOLD = 0.60
DEFECT_CONF_THRESHOLD = 0.25

CLASSIFICATION_IMAGE_SIZE = 224
DETECTION_IMAGE_SIZE = 640

MANGO_CLASS_NAMES = {
    "mango",
    "buah_mangga",
    "mango_fruit"
}

HEALTHY_CLASS_NAMES = {
    "healthy",
    "healthy_mango",
    "mango_healthy",
    "sehat"
}


# ==================================================
# NORMALISASI NAMA KELAS
# ==================================================

def normalize_class_name(class_name):
    """Menyamakan format nama kelas."""

    return (
        str(class_name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


# ==================================================
# MENCARI ID KELAS
# ==================================================

def find_class_id(names, target_names):
    """Mencari ID kelas berdasarkan nama kelas."""

    if isinstance(names, dict):
        class_items = names.items()
    else:
        class_items = enumerate(names)

    normalized_targets = {
        normalize_class_name(name)
        for name in target_names
    }

    for class_id, class_name in class_items:
        normalized_name = normalize_class_name(class_name)

        if normalized_name in normalized_targets:
            return int(class_id)

    return None


# ==================================================
# STATUS BOX RINGKAS
# Hanya dipakai untuk Not Mango atau tidak ada deteksi
# ==================================================

def draw_compact_status(img, text, status_type, conf=None):
    """
    Menggambar status kecil di kiri bawah gambar.

    Kotak ini tidak digunakan untuk hasil defect yang memiliki
    bounding box, sehingga tampilan tidak menutupi objek.
    """

    h, w = img.shape[:2]

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
    line_2 = None

    if conf is not None:
        line_2 = f"Confidence: {conf * 100:.1f}%"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale_title = 0.48
    font_scale_conf = 0.42
    thickness = 1

    (title_w, title_h), _ = cv2.getTextSize(
        line_1,
        font,
        font_scale_title,
        thickness
    )

    conf_w = 0
    conf_h = 0

    if line_2 is not None:
        (conf_w, conf_h), _ = cv2.getTextSize(
            line_2,
            font,
            font_scale_conf,
            thickness
        )

    padding_x = 12
    padding_y = 9
    gap = 6 if line_2 is not None else 0

    box_width = max(title_w, conf_w) + (padding_x * 2)
    box_height = title_h + conf_h + gap + (padding_y * 2)

    margin = 12

    # Pastikan kotak tidak melewati ukuran gambar
    box_width = min(box_width, max(120, w - (margin * 2)))

    x1 = margin
    y1 = max(margin, h - box_height - margin)

    x2 = min(w - margin, x1 + box_width)
    y2 = min(h - margin, y1 + box_height)

    overlay = img.copy()

    cv2.rectangle(
        overlay,
        (x1 + 3, y1 + 3),
        (x2 + 3, y2 + 3),
        (0, 0, 0),
        -1,
        cv2.LINE_AA
    )

    cv2.addWeighted(
        overlay,
        0.25,
        img,
        0.75,
        0,
        img
    )

    cv2.rectangle(
        img,
        (x1, y1),
        (x2, y2),
        bg_color,
        -1,
        cv2.LINE_AA
    )

    cv2.rectangle(
        img,
        (x1, y1),
        (x2, y2),
        border_color,
        1,
        cv2.LINE_AA
    )

    title_y = y1 + padding_y + title_h

    cv2.putText(
        img,
        line_1,
        (x1 + padding_x, title_y),
        font,
        font_scale_title,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )

    if line_2 is not None:
        conf_y = title_y + gap + conf_h

        cv2.putText(
            img,
            line_2,
            (x1 + padding_x, conf_y),
            font,
            font_scale_conf,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

    return img


# ==================================================
# DRAW SATU BOUNDING BOX TERBAIK
# Label dibuat kecil agar confidence tetap terlihat
# ==================================================

def draw_detection(img, box, names, color=(0, 140, 255)):
    """
    Menggambar satu bounding box dengan label kecil.

    Format label:
    nama_kelas 52.0%
    """

    h, w = img.shape[:2]

    coordinates = (
        box.xyxy[0]
        .detach()
        .cpu()
        .tolist()
    )

    x1, y1, x2, y2 = map(int, coordinates)

    # Membatasi koordinat agar tetap berada di dalam gambar
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))

    cls_id = int(box.cls[0].item())
    confidence = float(box.conf[0].item())
    class_name = str(names[cls_id])

    # Bounding box dibuat lebih tipis
    cv2.rectangle(
        img,
        (x1, y1),
        (x2, y2),
        color,
        2,
        cv2.LINE_AA
    )

    label_text = f"{class_name} | Conf: {confidence:.2f}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.42
    thickness = 1

    (text_width, text_height), baseline = cv2.getTextSize(
        label_text,
        font,
        font_scale,
        thickness
    )

    padding_x = 6
    padding_y = 4

    label_width = text_width + (padding_x * 2)
    label_height = text_height + baseline + (padding_y * 2)

    # Posisi horizontal disesuaikan agar label tidak terpotong
    label_x1 = x1

    if label_x1 + label_width > w:
        label_x1 = max(0, w - label_width)

    label_x2 = min(w - 1, label_x1 + label_width)

    # Label diletakkan di atas bounding box.
    # Jika tidak cukup ruang, label masuk ke bagian atas box.
    if y1 - label_height >= 0:
        label_y1 = y1 - label_height
        label_y2 = y1
        text_y = label_y2 - padding_y - baseline
    else:
        label_y1 = y1
        label_y2 = min(h - 1, y1 + label_height)
        text_y = label_y2 - padding_y - baseline

    cv2.rectangle(
        img,
        (label_x1, label_y1),
        (label_x2, label_y2),
        color,
        -1,
        cv2.LINE_AA
    )

    cv2.putText(
        img,
        label_text,
        (label_x1 + padding_x, text_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )

    return img


# ==================================================
# MENGAMBIL DETEKSI DENGAN CONFIDENCE TERTINGGI
# ==================================================

def get_best_detection(boxes, names):

    if boxes is None or len(boxes) == 0:
        return None, None, 0.0

    confidence_values = (
        boxes.conf
        .detach()
        .cpu()
        .numpy()
    )

    best_index = int(np.argmax(confidence_values))
    best_box = boxes[best_index]

    class_id = int(best_box.cls[0].item())
    confidence = float(best_box.conf[0].item())
    detected_class = names[class_id]

    return best_box, detected_class, confidence


# ==================================================
# KONVERSI GAMBAR KE RGB
# ==================================================

def prepare_image(image):
    """
    Menerima gambar PIL atau NumPy dan mengubahnya
    menjadi array RGB dengan tiga channel.
    """

    img = np.array(image)

    if img.ndim == 2:
        img = cv2.cvtColor(
            img,
            cv2.COLOR_GRAY2RGB
        )

    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(
            img,
            cv2.COLOR_RGBA2RGB
        )

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(
            "Format gambar tidak didukung. "
            "Gunakan gambar JPG, JPEG, atau PNG."
        )

    return img


# ==================================================
# MAIN DETECTION FUNCTION
# ==================================================

def detect_mango(
    image,
    conf_threshold=MANGO_CONF_THRESHOLD,
    defect_conf_threshold=DEFECT_CONF_THRESHOLD
):

    img_rgb = prepare_image(image)

    img_bgr = cv2.cvtColor(
        img_rgb,
        cv2.COLOR_RGB2BGR
    )

    output_img = img_bgr.copy()

    # ==================================================
    # STAGE 1: KLASIFIKASI MANGO / NON-MANGO
    # ==================================================

    classification_results = input_validator.predict(
    source=img_bgr,
    imgsz=CLASSIFICATION_IMAGE_SIZE,
    verbose=False
    )

    cls_result = classification_results[0]

    if cls_result.probs is None:
        raise RuntimeError(
            "Model klasifikasi tidak menghasilkan probabilitas."
        )

    class_names = cls_result.names

    mango_class_id = find_class_id(
        class_names,
        MANGO_CLASS_NAMES
    )

    if mango_class_id is None:
        raise ValueError(
            "Kelas 'mango' tidak ditemukan pada model klasifikasi. "
            "Pastikan dataset memiliki kelas 'mango' dan 'non_mango'."
        )

    top1_class_id = int(cls_result.probs.top1)
    top1_class_name = class_names[top1_class_id]

    normalized_top1_name = normalize_class_name(
        top1_class_name
    )

    mango_confidence = float(
        cls_result.probs.data[mango_class_id].item()
    )

    normalized_mango_names = {
        normalize_class_name(name)
        for name in MANGO_CLASS_NAMES
    }

    is_mango = (
        normalized_top1_name in normalized_mango_names
        and mango_confidence >= conf_threshold
    )

    # ==================================================
    # BUKAN MANGGA
    # ==================================================

    if not is_mango:

        output_img = draw_compact_status(
            output_img,
            "Not Mango",
            "not_mango",
            mango_confidence
        )

        status = (
            f"❌ Not Mango "
            f"({mango_confidence * 100:.1f}%)"
        )

        detected_class = "Not Mango"
        final_confidence = mango_confidence

    else:

        # ==================================================
        # STAGE 2: DETEKSI KONDISI MANGGA
        # ==================================================

        detection_results = defect_detector.predict(
            source=img_bgr,
            imgsz=DETECTION_IMAGE_SIZE,
            conf=defect_conf_threshold,
            verbose=False
        )

        det_result = detection_results[0]

        best_box, detected_class, defect_confidence = (
            get_best_detection(
                det_result.boxes,
                det_result.names
            )
        )

        # ==================================================
        # TIDAK ADA DETEKSI
        # ==================================================

        if best_box is None:

            output_img = draw_compact_status(
                output_img,
                "Healthy Mango",
                "healthy",
                mango_confidence
            )

            status = "✅ Healthy Mango"
            detected_class = "Healthy"
            final_confidence = mango_confidence

        else:

            normalized_detected_class = normalize_class_name(
                detected_class
            )

            normalized_healthy_names = {
                normalize_class_name(name)
                for name in HEALTHY_CLASS_NAMES
            }

            # ==================================================
            # HEALTHY TERDETEKSI
            # Hanya menampilkan bounding box kecil
            # ==================================================

            if normalized_detected_class in normalized_healthy_names:

                output_img = draw_detection(
                    output_img,
                    best_box,
                    det_result.names,
                    color=(0, 200, 0)
                )

                status = (
                    f"✅ Healthy Mango "
                    f"({defect_confidence * 100:.1f}%)"
                )

                detected_class = "Healthy"
                final_confidence = defect_confidence

            # ==================================================
            # CACAT TERDETEKSI
            # Hanya menampilkan bounding box, tanpa kotak status
            # ==================================================

            else:

                output_img = draw_detection(
                    output_img,
                    best_box,
                    det_result.names,
                    color=(0, 140, 255)
                )

                status = (
                    f"⚠️ Defect Mango: {detected_class} "
                    f"({defect_confidence * 100:.1f}%)"
                )

                final_confidence = defect_confidence

    output_img = cv2.cvtColor(
        output_img,
        cv2.COLOR_BGR2RGB
    )

    return (
        output_img,
        status,
        final_confidence,
        detected_class
    )
