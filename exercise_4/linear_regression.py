import numpy as np


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
