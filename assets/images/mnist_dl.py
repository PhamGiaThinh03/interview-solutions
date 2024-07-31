import numpy as np
import gzip
import struct
import os
import pickle

DATA_PATH = "/root/personal/dunglq12/practice-ds/interview_exercises_thinhpg/datasets"


def load_mnist_data(DATA_PATH, image_filename, label_filename):
    """Load MNIST data from the specified files in the base path."""
    image_path = os.path.join(DATA_PATH, image_filename)
    label_path = os.path.join(DATA_PATH, label_filename)

    # Load image data
    with gzip.open(image_path, "rb") as img_f:
        _, num_images, rows, cols = struct.unpack(">IIII", img_f.read(16))
        images = np.frombuffer(img_f.read(), dtype=np.uint8).reshape(
            num_images, rows * cols
        )
    # Load label data
    with gzip.open(label_path, "rb") as lbl_f:
        _, num_items = struct.unpack(">II", lbl_f.read(8))
        labels = np.frombuffer(lbl_f.read(), dtype=np.uint8)

    # Normalize images to the range [0, 1]
    return images / 255.0, labels


class SiameseNetwork:
    def __init__(self, input_dim, hidden_dim):
        """Initialize the Siamese Network with given dimensions."""
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.weights_1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.weights_2 = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.bias_1 = np.zeros(hidden_dim)
        self.bias_2 = np.zeros(hidden_dim)

    def forward(self, x):
        """Forward pass of the Siamese Network."""
        hidden = np.maximum(
            0, np.dot(x, self.weights_1) + self.bias_1
        )  # ReLU activation
        embedding = np.dot(hidden, self.weights_2) + self.bias_2
        return embedding

    def compute_triplet_loss(self, anchor, positive, negative, alpha=0.2):
        """Compute the triplet loss."""
        anchor_embedding = self.forward(anchor)
        positive_embedding = self.forward(positive)
        negative_embedding = self.forward(negative)

        # Compute pairwise distances
        d_ap = np.linalg.norm(anchor_embedding - positive_embedding, axis=1)
        d_an = np.linalg.norm(anchor_embedding - negative_embedding, axis=1)

        # Compute triplet loss
        loss = np.maximum(0, d_ap - d_an + alpha)
        return np.mean(loss)

    def train(self, X_train, y_train, epochs=10, alpha=0.2):
        """Train the Siamese Network using Triplet Loss."""
        num_samples = X_train.shape[0]
        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(num_samples):
                anchor = X_train[i]
                positive_indices = np.where(y_train == y_train[i])[0]
                negative_indices = np.where(y_train != y_train[i])[0]

                # Ensure there are enough positive and negative samples
                if len(positive_indices) > 1 and len(negative_indices) > 0:
                    positive_idx = np.random.choice(positive_indices)
                    negative_idx = np.random.choice(negative_indices)

                    positive = X_train[positive_idx]
                    negative = X_train[negative_idx]

                    # Compute triplet loss for the current sample
                    loss = self.compute_triplet_loss(
                        anchor[np.newaxis, :],
                        positive[np.newaxis, :],
                        negative[np.newaxis, :],
                        alpha,
                    )
                    epoch_loss += loss

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / num_samples}")

    def save_model(self, filename):
        """Save the model parameters to a .npz file."""
        np.savez_compressed(
            filename,
            weights_1=self.weights_1,
            weights_2=self.weights_2,
            bias_1=self.bias_1,
            bias_2=self.bias_2,
        )

    def load_model(self, filename):
        """Load model parameters from a .npz file."""
        data = np.load(filename)
        self.weights_1 = data["weights_1"]
        self.weights_2 = data["weights_2"]
        self.bias_1 = data["bias_1"]
        self.bias_2 = data["bias_2"]

    def predict(self, X):
        """Predict embeddings for the input data."""
        return self.forward(X)


def calculate_accuracy(X_test, y_test, model):
    """Calculate accuracy of the model on the test dataset."""
    num_samples = X_test.shape[0]
    correct = 0

    # Generate all embeddings
    embeddings = model.predict(X_test)

    for i in range(num_samples):
        distances = np.linalg.norm(embeddings[i] - embeddings, axis=1)
        nearest_neighbor = np.argmin(distances[1:]) + 1
        if y_test[i] == y_test[nearest_neighbor]:
            correct += 1

    return correct / num_samples


if __name__ == "__main__":

    X_train, y_train = load_mnist_data(
        DATA_PATH, "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"
    )
    X_test, y_test = load_mnist_data(
        DATA_PATH, "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"
    )

    input_dim = X_train.shape[1]
    hidden_dim = 128
    model = SiameseNetwork(input_dim, hidden_dim)
    model.train(X_train, y_train, epochs=10, alpha=0.2)

    model.save_model("siamese_model.npz")

    model.load_model("siamese_model.npz")

    accuracy = calculate_accuracy(X_test, y_test, model)
    print(f"Test accuracy: {accuracy:.4f}")
