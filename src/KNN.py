import numpy as np
from collections import Counter

def minkowski_distance(x, y, p=2):
    """
    Calculate the Minkowski distance between two points.

    Args:
        x (array-like): First point.
        y (array-like): Second point.
        p (int): The order of the Minkowski distance.

    Returns:
        float: The Minkowski distance between x and y.
    """
    return np.sum(np.abs(x - y) ** p) ** (1 / p)

class KDTree:
    """
    KD-Tree implementation for efficient nearest neighbor search.
    """
    def __init__(self, points, indices, depth=0):
        self.axis = depth % len(points[0])
        sorted_points = sorted(zip(points, indices), key=lambda x: x[0][self.axis])
        median_idx = len(sorted_points) // 2

        self.location, self.index = sorted_points[median_idx]
        self.left = KDTree(
            [p[0] for p in sorted_points[:median_idx]],
            [p[1] for p in sorted_points[:median_idx]],
            depth + 1
        ) if median_idx > 0 else None

        self.right = KDTree(
            [p[0] for p in sorted_points[median_idx + 1:]],
            [p[1] for p in sorted_points[median_idx + 1:]],
            depth + 1
        ) if median_idx + 1 < len(sorted_points) else None

    def nearest_neighbor(self, target, k=1, p=2):
        """
        Find the k nearest neighbors of the target point.

        Args:
            target (array-like): Target point.
            k (int): Number of neighbors to find.
            p (int): The order of the Minkowski distance.

        Returns:
            list: List of tuples (distance, index).
        """
        neighbors = []

        def search(tree, depth=0):
            if tree is None:
                return

            axis = depth % len(target)
            dist = minkowski_distance(target, tree.location, p)
            neighbors.append((dist, tree.index))
            neighbors.sort(key=lambda x: x[0])
            if len(neighbors) > k:
                neighbors.pop()

            diff = target[axis] - tree.location[axis]
            close, away = (tree.left, tree.right) if diff < 0 else (tree.right, tree.left)

            search(close, depth + 1)

            if len(neighbors) < k or abs(diff) < neighbors[-1][0]:
                search(away, depth + 1)

        search(self)
        return neighbors

class KNN:
    """
    K-Nearest Neighbors (KNN) classifier.
    """
    def __init__(self, k=3, p=2):
        """
        Initialize the KNN classifier.

        Args:
            k (int): Number of neighbors.
            p (int): The order of the Minkowski distance.
        """
        self.k = k
        self.p = p
        self.kdtree = None
        self.y_train = None

    def fit(self, X, y):
        """
        Fit the KNN classifier with training data.

        Args:
            X (array-like): Training features.
            y (array-like): Training labels.
        """
        self.kdtree = KDTree(list(X), list(range(len(X))))
        self.y_train = np.array(y)

    def predict(self, X):
        """
        Predict the labels for the given data points.

        Args:
            X (array-like): Test features.

        Returns:
            list: Predicted labels.
        """
        predictions = []
        for x in X:
            neighbors = self.kdtree.nearest_neighbor(x, self.k, self.p)
            indices = [neighbor[1] for neighbor in neighbors]
            labels = self.y_train[indices]
            most_common = Counter(labels).most_common(1)[0][0]
            predictions.append(most_common)
        return predictions