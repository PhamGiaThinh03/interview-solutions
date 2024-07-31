"""
    Bài 2: Triplet loss
        b. Việt công thức toán, implement (code numpy) và giải thích khi 
        Input triplet loss mở rộng không chỉ là 1 mẫu thật và một mẫu giả nữa mà sẽ là 2 mẫu thật và 5 mẫu giả.
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


def extended_triplet_loss(anchor, positives, negatives, margin=1.0):
    """
    Compute the extended triplet loss for a set of positive and negative samples.

    Args:
        anchor (np.ndarray): Embedding of the anchor sample.
        positives (list of np.ndarray): List of embeddings for positive samples (similar to anchor).
        negatives (list of np.ndarray): List of embeddings for negative samples (dissimilar to anchor).
        margin (float): Margin to ensure that the negative samples are at least this far from the anchor.

    Returns:
        float: Computed extended triplet loss.
    """
    positive_distances = np.array([euclidean_distance(anchor, p) for p in positives])
    negative_distances = np.array([euclidean_distance(anchor, n) for n in negatives])

    avg_positive_distance = np.mean(positive_distances)
    avg_negative_distance = np.mean(negative_distances)

    loss = np.maximum(0, avg_positive_distance - avg_negative_distance + margin)
    return loss


if __name__ == "__main":
    anchor = np.array([2.0, 7.0])
    positives = [np.array([1.1, 2.1]), np.array([1.2, 2.2])]
    negatives = [
        np.array([3.0, 3.0]),
        np.array([4.0, 4.0]),
        np.array([5.0, 5.0]),
        np.array([6.0, 6.0]),
        np.array([7.0, 7.0]),
    ]

    loss = extended_triplet_loss(anchor, positives, negatives)
    print(f"Extended triplet loss: {loss}")
