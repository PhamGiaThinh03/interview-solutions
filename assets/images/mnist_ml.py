"""
    Bài 1: Machine learning cơ bản
        Xây dựng một mô hình Machine learning (not deep learning)
        ứng dụng cho bài phân biệt loại ký tự quang học, ứng dụng data MNIST. Chỉ sử dụng numpy
"""

import numpy as np
import gzip
import struct
import pickle
import os

# Define the path where the MNIST dataset is located
DATA_PATH = "/root/personal/dunglq12/practice-ds/interview_exercise/datasets"


# Function to load MNIST data from the given files
def load_mnist_data(image_filename, label_filename):

    image_path = os.path.join(DATA_PATH, image_filename)
    label_path = os.path.join(DATA_PATH, label_filename)

    # Load and unpack the image file
    with gzip.open(image_path, "rb") as img_f:
        _, num_images, rows, cols = struct.unpack(">IIII", img_f.read(16))
        images = np.frombuffer(img_f.read(), dtype=np.uint8).reshape(
            num_images, rows * cols
        )
    # Load and unpack the label file
    with gzip.open(label_path, "rb") as lbl_f:
        _, num_items = struct.unpack(">II", lbl_f.read(8))
        labels = np.frombuffer(lbl_f.read(), dtype=np.uint8)

    # Normalize images to the range [0, 1]
    return images / 255.0, labels


# K-Nearest Neighbors classifier implementation
class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        distances = np.zeros((num_test, num_train))

        # Compute the distance from each test point to all training points
        for i in range(num_test):
            distances[i, :] = np.linalg.norm(self.X_train - X[i, :], axis=1)

        y_pred = np.zeros(num_test, dtype=int)
        # Find the k nearest neighbors and vote for the most common class
        for i in range(num_test):
            closest_y = self.y_train[np.argsort(distances[i, :])[: self.k]]
            y_pred[i] = np.bincount(closest_y).argmax()

        return y_pred


# Function to calculate the accuracy of predictions
def accuracy(predictions, labels):
    return np.mean(predictions == labels)


if __name__ == "__main__":

    X_train, y_train = load_mnist_data(
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
    )
    X_test, y_test = load_mnist_data(
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    )

    # Initialize and train the KNN classifier
    model = KNNClassifier(k=3)
    model.fit(X_train, y_train)

    pickle.dump(model, open("knn_model_v1.pkl", "wb"))

    # # Make predictions on the test data
    # test_predictions = model.predict(X_test)
    # print(f"Test accuracy: {accuracy(test_predictions, y_test):.4f}")
