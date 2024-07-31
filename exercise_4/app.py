import numpy as np
from linear_regression import LogisticRegression
from fastapi import FastAPI
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import cv2


def load_model(filename):
    """
    Load a pre-trained Logistic Regression model from a .npz file.

    Args:
        filename (str): Path to the .npz file containing the model parameters.

    Returns:
        LogisticRegression: An instance of the LogisticRegression class with loaded weights and biases.
    """
    data = np.load(filename)
    model = LogisticRegression(
        input_dim=data["weights"].shape[0], output_dim=data["weights"].shape[1]
    )
    model.weights = data["weights"]
    model.bias = data["bias"]
    return model


class PredictorAPI:
    def __init__(self):
        self.app = FastAPI()
        self.model = load_model("logistic_regression_model.npz")

        @self.app.post("/predict")
        async def predict(file: UploadFile = File(...)):
            try:
                # Read the uploaded image
                image_data = await file.read()
                image = np.array(Image.open(io.BytesIO(image_data)))

                image = cv2.resize(image, (28, 28))
                image = image.astype(float) / 255
                image = image.reshape(1, -1)
                prediction = self.model.predict(image)

                return JSONResponse(content={"prediction": prediction[0].tolist()})

            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))


api = PredictorAPI()
