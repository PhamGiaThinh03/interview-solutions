import numpy as np
from linear_regression import LogisticRegression


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


# Example usage
loaded_model = load_model("logistic_regression_model.npz")
