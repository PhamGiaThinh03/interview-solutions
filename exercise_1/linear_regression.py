import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import struct


class CFG:
    train_images_path = "../assets/MNIST/train-images-idx3-ubyte"
    train_labels_path = "../assets/MNIST/train-labels-idx1-ubyte"
    test_images_path = "../assets/MNIST/t10k-images-idx3-ubyte"
    test_labels_path = "../assets/MNIST/t10k-labels-idx1-ubyte"


def read_ubyte(file_name):
    """
    Read an MNIST data file in the IDX file format and return the data as a NumPy array.
    """
    with open(file_name, "rb") as file:
        _, _, dims = struct.unpack(">HBB", file.read(4))
        shape = tuple(struct.unpack(">I", file.read(4))[0] for _ in range(dims))
        return np.frombuffer(file.read(), dtype=np.uint8).reshape(shape)


def preprocess_data():
    """
    Preprocess the MNIST data by reading, normalizing, and one-hot encoding the labels.
    """
    x_train = read_ubyte(CFG.train_images_path)
    y_train = read_ubyte(CFG.train_labels_path)

    x_test = read_ubyte(CFG.test_images_path)
    y_test = read_ubyte(CFG.test_labels_path)

    x_train = x_train / 255.0
    x_test = x_test / 255.0
    x_train = x_train.reshape(-1, 28 * 28)
    x_test = x_test.reshape(-1, 28 * 28)

    # One-hot encode labels
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]

    return (x_train, y_train), (x_test, y_test)


class LogisticRegression:
    """
    Initialize the Logistic Regression model with weights and biases.
    """

    def __init__(self, input_dim, output_dim, learning_rate=0.02, epochs=500):
        self.weights = np.random.randn(input_dim, output_dim) * 0.01
        self.bias = np.zeros(output_dim)
        self.learning_rate = learning_rate
        self.epochs = epochs

    def softmax(self, z):
        """
        Compute the softmax of the input array.
        """
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, x):
        """
        Perform the forward pass of the model.
        """
        return self.softmax(np.dot(x, self.weights) + self.bias)

    def compute_loss(self, y_true, y_pred):
        """
        Compute the cross-entropy loss between true labels and predictions.
        """
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))

    def backward(self, x, y_true, y_pred):
        """
        Perform the backward pass to compute gradients of the loss function.
        """
        m = x.shape[0]
        dz = (y_pred - y_true) / m
        dw = np.dot(x.T, dz)
        db = np.sum(dz, axis=0)
        return dw, db

    def fit(self, x_train, y_train):
        """
        Train the logistic regression model using gradient descent.
        """
        for epoch in range(self.epochs):
            y_pred = self.forward(x_train)
            loss = self.compute_loss(y_train, y_pred)
            dw, db = self.backward(x_train, y_train, y_pred)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, x):
        """
        Predict the class labels for the given data.
        """
        y_pred = self.forward(x)
        return np.argmax(y_pred, axis=1)


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = preprocess_data()

    model = LogisticRegression(
        input_dim=x_train.shape[1],
        output_dim=y_train.shape[1],
        learning_rate=0.02,
        epochs=500,
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(np.argmax(y_test, axis=1), y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    def save_model(model, filename):
        np.savez(filename, weights=model.weights, bias=model.bias)

    # Example usage
    save_model(model, "logistic_regression_model.npz")
