import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from .model_loader import load_model

router = APIRouter()
model = load_model()

# 5 MB - very arbitrary number.
MAX_FILE_SIZE = 5 * 1024 * 1024


@router.post("/detect/")
async def detect(file: UploadFile = File(...)):
    """
    Detect items on a given uploaded image.
    """
    content = await file.read()

    if file.content_type != "image/jpeg":
        raise HTTPException(status_code=415, detail="invalid format")

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="file too large"
        )

    image_array = np.frombuffer(content, np.uint8)

    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    results = model(image)
    detections = results[0].boxes

    detected_objects = []

    for box in detections:
        confidence = float(box.conf[0])

        class_index = int(box.cls[0])
        class_name = results[0].names[class_index]

        detected_objects.append({
            "name": class_name,
            "confidence": confidence
        })

    return {
        "detected_objects": detected_objects
    }
