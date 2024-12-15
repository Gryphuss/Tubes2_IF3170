import numpy as np
import pandas as pd
from math import sqrt, pi, exp
from typing import Dict, Any, Union

class GaussianNaiveBayes:
    def __init__(self) -> None:
        self.classes: np.ndarray | None = None  
        self.mean: Dict[Union[int, float], np.ndarray] = {} 
        self.variance: Dict[Union[int, float], np.ndarray] = {}  
        self.priors: Dict[Union[int, float], float] = {}  

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Validate input dimensions
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in features (X) and labels (y) must match.")

        self.classes = np.unique(y)
        for cls in self.classes:
            X_cls = X[y == cls]
            self.mean[cls] = X_cls.mean(axis=0)
            self.variance[cls] = X_cls.var(axis=0) 
            self.priors[cls] = X_cls.shape[0] / X.shape[0]

    def _calculate_likelihood(self, x: float, mean: float, variance: float) -> float:
        eps = 1e-6  # To avoid division by zero
        coeff = 1 / sqrt(2 * pi * (variance + eps))
        exponent = exp(-((x - mean) ** 2) / (2 * (variance + eps)))
        return coeff * exponent

    def _calculate_posterior(self, x: np.ndarray) -> Dict[Union[int, float], float]:
        posteriors: Dict[Union[int, float], float] = {}
        for cls in self.classes:
            prior: float = np.log(self.priors[cls])   # Use log to prevent underflow
            likelihood: float = np.sum(np.log(self._calculate_likelihood(x, self.mean[cls], self.variance[cls])))
            posteriors[cls] = prior + likelihood
        return posteriors

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred: list[Any] = []
        for x in X:
            posteriors = self._calculate_posterior(x)
            y_pred.append(max(posteriors, key=posteriors.get))
        return np.array(y_pred)

