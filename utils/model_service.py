from functools import lru_cache

from ultralytics import YOLO


@lru_cache(maxsize=1)
def load_models():
    input_validator = YOLO("model/best_cls.pt")
    defect_detector = YOLO("model/best.pt")

    return input_validator, defect_detector