import io
from fastapi.testclient import TestClient
from .detection import router

client = TestClient(router)


def test_success_multiple_detections():
    """
    Test that multiple objects are successfully detected in an image.
    """
    with open("test_data/shelf.jpg", "rb") as test_image:
        files = {'file': ('shelf.jpg', test_image, 'image/jpeg')}
        response = client.post("/detect/", files=files)

    for obj in response.json()["detected_objects"]:
        assert "name" in obj
        assert "confidence" in obj
        assert isinstance(obj["name"], str)
        assert isinstance(obj["confidence"], float)
        assert 0 <= obj["confidence"] <= 1

    assert response.status_code == 200


def test_success_single_detection():
    """
    Test that a single object is successfully detected in an image.
    """
    with open("test_data/wineglass.jpg", "rb") as test_image:
        files = {'file': ('wineglass.jpg', test_image, 'image/jpeg')}
        response = client.post("/detect/", files=files)

    detection = [obj["name"] for obj in response.json()["detected_objects"]]

    assert response.status_code == 200
    assert "wine glass" in detection


def test_invalid_format():
    """
    Test that uploading a file with an invalid format is rejected with 415.
    """
    with open("test_data/textfile.txt", "rb") as file:
        response = client.post(
            "/detect/", files={"file": ("textfile.txt", file, "text/plain")})

    assert response.status_code == 415


def test_no_file():
    """
    Test that an empty request without a file is rejected with 422
    """
    response = client.post("/detect/", files={})

    assert response.status_code == 422


def test_large_file_too_big():
    """
    Test that uploading a file larger than the maximum allowed size is rejected.
    """
    large_file_size = 6 * 1024 * 1024  # 6MB
    large_file_content = b'a' * large_file_size  # Create 6MB of content
    file_like_object = io.BytesIO(large_file_content)

    response = client.post(
        "/detect/", files={"file": ("large_image.jpg", file_like_object, "image/jpeg")})

    assert response.status_code == 413
    assert response.json()[
        "detail"] == "File size exceeds the maximum limit of 5.0 MB"
