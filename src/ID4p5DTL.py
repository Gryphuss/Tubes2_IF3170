import numpy as np
import pandas as pd
from collections import Counter
from math import log2


### NOT WORKING YET
class ID4p5DecisionTreeWithDynamicPruning:
    def __init__(self, base_min_samples_split=2, base_min_gain=0.01):
        self.tree = None
        self.base_min_samples_split = base_min_samples_split
        self.base_min_gain = base_min_gain
        self.dynamic_pruning_factor = 1.0

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
        if X.shape[1] == 0 or len(y) < self.dynamic_min_samples_split:
            return Counter(y).most_common(1)[0][0]

        best_feature_idx = self._best_split(X, y)
        if best_feature_idx is None or self._information_gain(X[:, best_feature_idx], y) < self.dynamic_min_gain:
            return Counter(y).most_common(1)[0][0]

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

    def _prune(self, tree, X, y, feature_names):
        if not isinstance(tree, dict):
            return tree

        feature = next(iter(tree))
        feature_idx = feature_names.index(feature)

        for value in list(tree[feature].keys()):
            subtree = tree[feature][value]

            branch_X = X[X[:, feature_idx] == value]
            branch_y = y[X[:, feature_idx] == value]

            if isinstance(subtree, dict):
                tree[feature][value] = self._prune(subtree, branch_X, branch_y, feature_names)

        if all(not isinstance(subtree, dict) for subtree in tree[feature].values()):
            leaf_label = Counter(y).most_common(1)[0][0]
            original_accuracy = self._evaluate(tree, X, y, feature_names)
            pruned_accuracy = self._evaluate(leaf_label, X, y, feature_names)

            if pruned_accuracy >= original_accuracy:
                print(f"Pruned feature: {feature}, replaced with label: {leaf_label}")
                return leaf_label

        return tree

    def _evaluate(self, model, X, y, feature_names):
        predictions = self.predict(X, feature_names, model)
        return np.mean(predictions == y)

    def fit(self, X, y, feature_names, validation_data=None):
        X = np.array(X)
        y = np.array(y)

        data_size = len(y)
        self.dynamic_pruning_factor = min(1.0, data_size / 1000)
        self.dynamic_min_samples_split = max(int(self.base_min_samples_split * self.dynamic_pruning_factor), 2)
        self.dynamic_min_gain = self.base_min_gain * self.dynamic_pruning_factor

        print(f"Dynamic Pruning Factor: {self.dynamic_pruning_factor}")
        print(f"Dynamic Min Samples Split: {self.dynamic_min_samples_split}")
        print(f"Dynamic Min Gain: {self.dynamic_min_gain}")

        self.tree = self._build_tree(X, y, feature_names)

        if validation_data:
            X_val, y_val = validation_data
            print("Applying post-pruning...")
            self.tree = self._prune(self.tree, X_val, y_val, feature_names)

    def predict(self, X, feature_names, tree=None):
        if tree is None:
            tree = self.tree
        predictions = []
        for sample in X:
            sample_dict = dict(zip(feature_names, sample))
            predictions.append(self._predict_single(tree, sample_dict))
        return np.array(predictions)

    def _predict_single(self, tree, sample):
        if not isinstance(tree, dict):
            return tree
        feature = next(iter(tree))
        value = sample.get(feature, None)
        subtree = tree.get(feature, {}).get(value, None)
        if subtree is None:
            return None
        return self._predict_single(subtree, sample)

    def print_tree(self):
        import pprint
        pprint.pprint(self.tree)


if __name__ == "__main__":
    data = {
        "Outlook": ["Sunny", "Sunny", "Overcast", "Rain", "Rain", "Rain", "Overcast", "Sunny", "Sunny", "Rain"],
        "Temperature": ["Hot", "Hot", "Hot", "Mild", "Cool", "Cool", "Cool", "Mild", "Cool", "Mild"],
        "Humidity": ["High", "High", "High", "High", "Normal", "Normal", "Normal", "High", "Normal", "Normal"],
        "Wind": ["Weak", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Weak", "Weak"],
        "PlayTennis": ["No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes"]
    }

    df = pd.DataFrame(data)
    feature_names = list(df.columns[:-1])
    X = df[feature_names].values
    y = df["PlayTennis"].values

    validation_data = (X[7:], y[7:])

    tree = ID4p5DecisionTreeWithDynamicPruning(base_min_samples_split=2, base_min_gain=0.01)
    tree.fit(X[:7], y[:7], feature_names, validation_data)
    print("awdawd")
    tree.print_tree()
    print("awdawd")


    test_samples = [["Sunny", "Cool", "High", "Strong"]]
    predictions = tree.predict(test_samples, feature_names)
    print("Predictions:", predictions)

