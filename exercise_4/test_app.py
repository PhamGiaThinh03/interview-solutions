import pytest
from fastapi.testclient import TestClient
from app import api  # Ensure you import the FastAPI app from your main.py

client = TestClient(api.app)


@pytest.fixture
def sample_image():
    """Create a sample image to use for testing."""
    from PIL import Image
    import io
    import numpy as np

    # Create an image with RGB channels
    image = Image.open("img_1.jpg")

    # Save it to a bytes buffer
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    return buffer


def test_predict(sample_image):
    """Test the /predict endpoint with a sample image."""
    response = client.post(
        "/predict", files={"file": ("test_image.png", sample_image, "image/png")}
    )

    assert response.status_code == 200
    assert "prediction" in response.json()


def test_predict_invalid_file():
    """Test the /predict endpoint with an invalid file."""
    response = client.post(
        "/predict", files={"file": ("test_image.txt", b"not an image", "text/plain")}
    )

    assert response.status_code == 400
    assert "detail" in response.json()
