"""
    Bài 2: Triplet loss
        a. Viết công thức tóan, implement (code numpy) và giải thích về Triplet loss.
"""

import numpy as np


def euclidean_distance(x1, x2):
    """
    Calculate the Euclidean distance between two points.

    Args:
        x1 (np.ndarray): Coordinates of the first point.
        x2 (np.ndarray): Coordinates of the second point.

    Returns:
        float: Euclidean distance between the two points.
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))


def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Compute the triplet loss between an anchor, positive, and negative sample.

    Args:
        anchor (np.ndarray): Embedding of the anchor point.
        positive (np.ndarray): Embedding of the positive point (similar to anchor).
        negative (np.ndarray): Embedding of the negative point (dissimilar to anchor).
        margin (float): Margin to ensure the negative pair is at least this far from the anchor.

    Returns:
        float: Computed triplet loss.
    """
    positive_distance = euclidean_distance(anchor, positive)
    negative_distance = euclidean_distance(anchor, negative)
    loss = np.maximum(0, positive_distance - negative_distance + margin)
    return loss


# Example data points:
anchor = np.array([2.0, 7.0])
positive = np.array([1.1, 2.1])
negative = np.array([3.0, 3.0])

# Calculate and print the triplet loss for the example data points
loss = triplet_loss(anchor, positive, negative)
print(f"Triplet loss: {loss}")
