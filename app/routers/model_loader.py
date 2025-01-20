import os
from ultralytics import YOLO


def load_model():
    model_name = os.getenv("MODEL_NAME", "yolov8n.pt")
    model_path = f"models/{model_name}"
    try:
        model = YOLO(model_path)
        return model
    except Exception:
        raise
