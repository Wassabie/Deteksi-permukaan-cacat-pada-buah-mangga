from ultralytics import YOLO
import cv2
import numpy as np


# ==================================================
# LOAD MODEL
# ==================================================

model_cls = YOLO("model/best_cls.pt")
model_det = YOLO("model/best.pt")


# ==================================================
# DRAW STATUS BOX
# ==================================================

def draw_status(img, text, status_type, conf=None):

    h, w = img.shape[:2]

    if status_type == "not_mango":
        color = (0, 0, 255)
        bg_color = (0, 0, 180)
        icon = "X"

    elif status_type == "healthy":
        color = (0, 200, 0)
        bg_color = (0, 130, 0)
        icon = "OK"

    else:
        color = (0, 140, 255)
        bg_color = (0, 90, 180)
        icon = "!"

    box_width = 320
    box_height = 75
    margin = 20

    x1 = margin
    y1 = h - box_height - margin

    x2 = x1 + box_width
    y2 = y1 + box_height

    overlay = img.copy()

    cv2.rectangle(
        overlay,
        (x1 + 4, y1 + 4),
        (x2 + 4, y2 + 4),
        (0, 0, 0),
        -1,
        cv2.LINE_AA
    )

    cv2.addWeighted(
        overlay,
        0.35,
        img,
        0.65,
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
        color,
        2,
        cv2.LINE_AA
    )

    cv2.rectangle(
        img,
        (x1, y1),
        (x1 + 8, y2),
        (255, 255, 255),
        -1
    )

    cv2.putText(
        img,
        f"{icon} {text}",
        (x1 + 20, y1 + 30),
        cv2.FONT_HERSHEY_DUPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    if conf is not None:
        cv2.putText(
            img,
            f"Confidence: {conf:.2f}",
            (x1 + 20, y1 + 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    return img


# ==================================================
# DRAW DEFECT DETECTION
# ==================================================

def draw_detection(img, boxes, names):

    for box in boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = names[cls]

        color = (0, 140, 255)

        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            color,
            2,
            cv2.LINE_AA
        )

        label_text = f"{label} {conf:.2f}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 2

        (text_w, text_h), baseline = cv2.getTextSize(
            label_text,
            font,
            font_scale,
            thickness
        )

        label_y = y1 - 12

        if label_y < 25:
            label_y = y1 + 30

        cv2.rectangle(
            img,
            (x1, label_y - text_h - 10),
            (x1 + text_w + 14, label_y + baseline - 2),
            color,
            -1,
            cv2.LINE_AA
        )

        cv2.rectangle(
            img,
            (x1, label_y - text_h - 10),
            (x1 + text_w + 14, label_y + baseline - 2),
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            img,
            label_text,
            (x1 + 7, label_y - 5),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

    return img


# ==================================================
# GET BEST DEFECT CLASS
# ==================================================

def get_best_detection(boxes, names):

    if len(boxes) == 0:
        return "Healthy", 0.0

    best_box = max(
        boxes,
        key=lambda box: float(box.conf[0])
    )

    cls_id = int(best_box.cls[0])
    conf = float(best_box.conf[0])
    detected_class = names[cls_id]

    return detected_class, conf


# ==================================================
# MAIN DETECTION FUNCTION
# ==================================================

def detect_mango(image, conf_threshold=0.6):

    img = np.array(image)

    img = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2BGR
    )

    # ==================================================
    # STAGE 1 : CLASSIFICATION MANGO / NON-MANGO
    # ==================================================

    cls_result = model_cls(img)[0]

    cls_name = cls_result.names[
        cls_result.probs.top1
    ]

    cls_conf = float(
        cls_result.probs.top1conf
    )

    detected_class = "-"

    # ==================================================
    # NOT MANGO
    # ==================================================

    if cls_name == "non_mango" or cls_conf < conf_threshold:

        img = draw_status(
            img,
            "Not Mango",
            "not_mango",
            cls_conf
        )

        status = f"❌ Not Mango ({cls_conf:.2f})"
        detected_class = "Not Mango"

    else:

        # ==================================================
        # STAGE 2 : OBJECT DETECTION DEFECT
        # ==================================================

        results = model_det(img)[0]

        # ==================================================
        # HEALTHY MANGO
        # ==================================================

        if len(results.boxes) == 0:

            img = draw_status(
                img,
                "Healthy Mango",
                "healthy",
                cls_conf
            )

            status = "✅ Healthy Mango"
            detected_class = "Healthy"

        # ==================================================
        # DEFECT MANGO
        # ==================================================

        else:

            detected_class, defect_conf = get_best_detection(
                results.boxes,
                model_det.names
            )

            img = draw_detection(
                img,
                results.boxes,
                model_det.names
            )

            img = draw_status(
                img,
                f"Defect: {detected_class}",
                "defect",
                defect_conf
            )

            status = "⚠️ Defect Mango"
            cls_conf = defect_conf

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    return img, status, cls_conf, detected_class