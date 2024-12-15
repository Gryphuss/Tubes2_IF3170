import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import OneHotEncoder

class KNNben:
    def __init__(self, k=3, p=2, categorical_columns=None):
        """
        Initializes the KNNben classifier.
        k: The number of nearest neighbors to consider.
        p: The power parameter for the Minkowski distance.
        categorical_columns: List of column names for categorical features.
        """
        self.k = k
        self.p = p
        self.categorical_columns = categorical_columns
    
    def fit(self, X_train, y_train):
        """
        Train the model using the provided training data.
        X_train: A pandas DataFrame containing numerical and categorical features.
        y_train: A pandas Series containing labels.
        """
        # Handle categorical data by one-hot encoding
        if self.categorical_columns:
            self.encoder = OneHotEncoder(sparse_output=False)
            categorical_data = X_train[self.categorical_columns]
            one_hot_encoded = self.encoder.fit_transform(categorical_data)
            
            # Drop the categorical columns and add the one-hot encoded columns
            X_train = X_train.drop(columns=self.categorical_columns)
            X_train = pd.concat([X_train, pd.DataFrame(one_hot_encoded)], axis=1)
        
        self.X_train = X_train.values
        self.y_train = y_train
    
    def predict(self, X_test):
        """
        Predict the labels for the test data.
        X_test: A pandas DataFrame containing the test data.
        
        Returns a 1D array of predicted labels.
        """
        # Handle categorical data in the test set by one-hot encoding
        if self.categorical_columns:
            categorical_data = X_test[self.categorical_columns]
            one_hot_encoded = self.encoder.transform(categorical_data)
            
            # Drop the categorical columns and add the one-hot encoded columns
            X_test = X_test.drop(columns=self.categorical_columns)
            X_test = pd.concat([X_test, pd.DataFrame(one_hot_encoded)], axis=1)
        
        predictions = [self._predict(x) for x in X_test.values]
        return np.array(predictions)
    
    def predict_proba(self, X_test):
        """
        Predict the probabilities of each class for the test data.
        X_test: A pandas DataFrame containing the test data.
        
        Returns a 2D array of probabilities for each class.
        """
        # Handle categorical data in the test set by one-hot encoding
        if self.categorical_columns:
            categorical_data = X_test[self.categorical_columns]
            one_hot_encoded = self.encoder.transform(categorical_data)
            
            # Drop the categorical columns and add the one-hot encoded columns
            X_test = X_test.drop(columns=self.categorical_columns)
            X_test = pd.concat([X_test, pd.DataFrame(one_hot_encoded)], axis=1)
        
        probas = [self._predict_proba(x) for x in X_test.values]
        return np.array(probas)
    
    def _predict(self, x):
        """
        Predict the label for a single test point `x`.
        """
        # Compute distances from x to all training points using Minkowski distance
        distances = [self._minkowski_distance(x, x_train) for x_train in self.X_train]
        
        # Get the indices of the k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        # print(k_indices)
        # Get the labels of the k nearest neighbors
        k_nearest_labels = [self.y_train.iloc[i] for i in k_indices]
        
        # Return the most common class label among the k nearest neighbors
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    
    def _predict_proba(self, x):
        """
        Predict the probabilities for a single test point `x`.
        """
        # Compute distances from x to all training points using Minkowski distance
        distances = [self._minkowski_distance(x, x_train) for x_train in self.X_train]
        
        # Get the indices of the k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # Get the labels of the k nearest neighbors
        k_nearest_labels = [self.y_train.iloc[i] for i in k_indices]
        
        # Calculate the proportion of each class in the k nearest neighbors
        class_counts = Counter(k_nearest_labels)
        total_neighbors = self.k
        probas = {cls: count / total_neighbors for cls, count in class_counts.items()}
        
        # Ensure all classes are included in the probability distribution
        all_classes = np.unique(self.y_train)
        probas_array = np.array([probas.get(cls, 0) for cls in all_classes])
        
        return probas_array
    
    def _minkowski_distance(self, x1, x2):
        """
        Calculate the Minkowski distance between two points x1 and x2.
        p: The power parameter for the Minkowski distance.
        """
        return np.sum(np.abs(x1 - x2) ** self.p) ** (1 / self.p)

# Example usage:
if __name__ == "__main__":
    # Example training data with numerical and categorical columns
    data = {
        'numerical1': [1, 2, 3, 4, 5, 6],
        'categorical1': ['A', 'B', 'A', 'C', 'B', 'A'],
        'categorical2': ['X', 'Y', 'X', 'Y', 'X', 'X']
    }
    X_train = pd.DataFrame(data)
    y_train = pd.Series([0, 0, 1, 1, 0, 1])

    # Example test data
    X_test = pd.DataFrame({
        'numerical1': [2, 5],
        'categorical1': ['A', 'C'],
        'categorical2': ['X', 'Y']
    })

    # Categorical columns names
    categorical_columns = ['categorical1', 'categorical2']

    # Instantiate and train the kNNben classifier with Minkowski distance (p=3)
    knn = KNNben(k=3, p=3, categorical_columns=categorical_columns)
    knn.fit(X_train, y_train)

    # Predict the labels for the test data
    predictions = knn.predict(X_test)
    print(f"Predictions: {predictions}")

    # Predict the probabilities for the test data
    probas = knn.predict_proba(X_test)
    print(f"Probabilities: {probas}")
