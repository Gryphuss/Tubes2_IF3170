import numpy as np
import pandas as pd
from collections import Counter
from math import log2


class ID3DecisionTree:
    def __init__(self):
        self.tree = None

    def _entropy(self, y):
        total = len(y)
        counts = Counter(y)
        return -sum((count / total) * log2(count / total) for count in counts.values() if count > 0)

    def _information_gain(self, X_col, y):
        total_entropy = self._entropy(y)
        total_samples = len(X_col)

        unique_values, counts = np.unique(X_col, return_counts=True)
        weighted_entropy = sum(
            (counts[i] / total_samples) * self._entropy(y[X_col == unique_values[i]])
            for i in range(len(unique_values))
        )

        return total_entropy - weighted_entropy

    def _best_split(self, X, y):
        best_gain = -1
        best_feature = None

        for col_idx in range(X.shape[1]):
            gain = self._information_gain(X[:, col_idx], y)
            if gain > best_gain:
                best_gain = gain
                best_feature = col_idx

        return best_feature

    def _build_tree(self, X, y, feature_names):
        if len(set(y)) == 1:
            return y[0]
        if X.shape[1] == 0:
            return Counter(y).most_common(1)[0][0]

        best_feature_idx = self._best_split(X, y)
        best_feature_name = feature_names[best_feature_idx]

        tree = {best_feature_name: {}}
        feature_values = np.unique(X[:, best_feature_idx])

        for value in feature_values:
            sub_X = X[X[:, best_feature_idx] == value]
            sub_y = y[X[:, best_feature_idx] == value]
            sub_feature_names = np.delete(feature_names, best_feature_idx)

            tree[best_feature_name][value] = self._build_tree(
                np.delete(sub_X, best_feature_idx, axis=1), sub_y, sub_feature_names
            )

        return tree

    def fit(self, X, y, feature_names):
        X = np.array(X)
        y = np.array(y)
        self.tree = self._build_tree(X, y, feature_names)

    def _predict_single(self, tree, sample):
        if not isinstance(tree, dict):
            return tree

        feature = next(iter(tree))
        value = sample[feature]

        subtree = tree.get(feature, {}).get(value, None)
        if subtree is None:
            return None
        return self._predict_single(subtree, sample)

    def predict(self, X, feature_names):
        predictions = []
        for sample in X:
            sample_dict = dict(zip(feature_names, sample))
            predictions.append(self._predict_single(self.tree, sample_dict))
        return np.array(predictions)

    def print_tree(self):
        import pprint
        pprint.pprint(self.tree)

if __name__ == "__main__":
    data = {
        "Outlook": ["Sunny", "Sunny", "Overcast", "Rain", "Rain", "Rain", "Overcast", "Sunny", "Sunny", "Rain", "Sunny", "Overcast", "Overcast", "Rain"],
        "Temperature": ["Hot", "Hot", "Hot", "Mild", "Cool", "Cool", "Cool", "Mild", "Cool", "Mild", "Mild", "Mild", "Hot", "Mild"],
        "Humidity": ["High", "High", "High", "High", "Normal", "Normal", "Normal", "High", "Normal", "Normal", "Normal", "High", "Normal", "High"],
        "Wind": ["Weak", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Strong"],
        "PlayTennis": ["No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"]
    }

    df = pd.DataFrame(data)
    feature_names = list(df.columns[:-1])
    X = df[feature_names].values
    y = df["PlayTennis"].values

    tree = ID3DecisionTree()
    tree.fit(X, y, feature_names)
    tree.print_tree()

    test_samples = [
        ["Sunny", "Cool", "High", "Strong"],
        ["Rain", "Mild", "Normal", "Weak"]
    ]
    predictions = tree.predict(test_samples, feature_names)
    print("Predictions:", predictions)
