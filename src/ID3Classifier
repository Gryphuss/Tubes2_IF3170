import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Any, Optional

class ID3Classifier:
    def __init__(self, max_depth: int = None):
        self.tree = None
        self.max_depth = max_depth
        self.feature_names = None
        self.target_column = None
        self.class_labels = None

    def _entropy(self, labels):
        label_counts = Counter(labels)
        probabilities = np.array([count / len(labels) for count in label_counts.values()])
        return -np.sum(probabilities * np.log2(probabilities))

    def _information_gain(self, data, feature_idx, labels):
        total_entropy = self._entropy(labels)
        unique_values, counts = np.unique(data[:, feature_idx], return_counts=True)
        weighted_entropy = 0
        for value, count in zip(unique_values, counts):
            subset = labels[data[:, feature_idx] == value]
            weighted_entropy += (count / len(data)) * self._entropy(subset)
        return total_entropy - weighted_entropy

    def _most_common_label(self, labels):
        return Counter(labels).most_common(1)[0][0]

    def _build_tree(self, data, features, labels, depth=0):
        if len(np.unique(labels)) == 1 or len(features) == 0 or \
           (self.max_depth is not None and depth >= self.max_depth):
            return self._most_common_label(labels)

        gains = [self._information_gain(data, f, labels) for f in features]
        best_feature_idx = features[np.argmax(gains)]

        tree = {best_feature_idx: {}}
        for value in np.unique(data[:, best_feature_idx]):
            mask = data[:, best_feature_idx] == value
            subtree = self._build_tree(data[mask], [f for f in features if f != best_feature_idx], labels[mask], depth + 1)
            tree[best_feature_idx][value] = subtree
        return tree

    def fit(self, X, y):
        self.feature_names = np.arange(X.shape[1])  # Use column indices as feature names
        self.tree = self._build_tree(X, self.feature_names, y)
        return self

    def predict(self, X):
        return np.array([self._predict_sample(sample) for sample in X])

    def _predict_sample(self, sample):
        node = self.tree
        while isinstance(node, dict):
            feature = list(node.keys())[0]

            value = sample[feature]

            if value not in node[feature]:
                return self._probabilistic_prediction(sample)

            node = node[feature][value]

        return node

    def _probabilistic_prediction(self, sample):
        all_labels = self._extract_labels(self.tree)

        if not all_labels:
            return np.random.choice(self.class_labels)

        return np.random.choice(all_labels)

    def _extract_labels(self, tree):
        labels = []
        def _extract(node):
            if isinstance(node, dict):
                for subnode in node.values():
                    _extract(subnode)
            else:
                labels.append(node)
        _extract(tree)
        return labels
