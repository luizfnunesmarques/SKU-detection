import os
from ultralytics import YOLO

MODEL_NAME = os.getenv('MODEL_NAME', 'yolov8n.pt')
MODEL_PATH = os.getenv('MODEL_PATH', '/models/' + MODEL_NAME)


def download_yolo_model():
    model_path = os.path.join(MODEL_PATH)

    try:
        YOLO(model_path)
        print(
            f"YOLO model '{MODEL_NAME}' successfully saved at '{model_path}'.")
    except Exception as e:
        print(e)


if __name__ == "__main__":
    download_yolo_model()
