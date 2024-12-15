import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k: int = 3, n: int = 2):
        self.k = k
        self.n = n

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        if not self._is_numerical(X):
            raise ValueError("The training data (X) must be numerical.")
        if not self._is_numerical(y):
            raise ValueError("The training labels (y) must be numerical.")
        
        self.X_train = X
        self.y_train = y

    def _is_numerical(self, data):
        try:
            np.asarray(data, dtype=np.float64)
            return True
        except ValueError:
            return False

    def _euclidean_distance(self, x1, x2):
        if self.n % 2 == 0:
            return (np.sum((x1 - x2) ** self.n)) ** (1 / self.n)
        else:
            return (np.sum(abs((x1 - x2) ** self.n))) ** (1 / self.n)

    def predict(self, X):
        X = np.array(X)
        if not self._is_numerical(X):
            raise ValueError("The test data (X) must be numerical.")
        
        predictions = [self.__predict_single(x) for x in X]
        return np.array(predictions)

    def __predict_single(self, x):
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_labels).most_common(1)
        return most_common[0][0]

# Example:
if __name__ == "__main__":
    X_train = [[1, 2], [2, 3], [3, 3], [6, 8], [7, 9], [8, 8]]
    y_train = [0, 0, 0, 1, 1, 1]

    X_test = [[2, 2], [7, 7], [5, 5], [1, 2], [1,1], [1,1]]

    knn = KNN()
    try:
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)
        print("Predictions:", predictions)
    except ValueError as e:
        print(f"Error: {e}")
