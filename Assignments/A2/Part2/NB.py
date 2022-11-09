__author__ = ["Shawn Santhoshgeorge (1006094673)", "Anaqi Amir Razif (1005813880)"]

from helper import *

import numpy as np

class NB:
    """
    Creates a Naive Bayes Classifier
    """

    def __init__(self) -> None:
        """
        Initializes a Naive Bayes Classifier
        """

        self.a = np.ndarray           # Likelihoods
        self.b = np.ndarray           # Class Priors

    def train(self, X: np.ndarray, y: np.ndarray, labels: list[str],) -> None:
        """
        Calculates the Model Likelihoods and Class Priors

        Args:
            X  (np.ndarray (Shape: (N, 9635))): A array consisting of each article with the usage of 9635 terms
            y  (np.ndarray (Shape: (N, 1))): A array consisting of the label index for the related article
            label (str[]): Classes for each unique y
        """

        assert X.shape[0] == y.shape[0], f"Number of inputs and outputs are different. (X: {X.shape[0]}, y: {y.shape[0]})"
        assert X.shape[1] == 9635, f"Each input should contain two components. Got: {X.shape[1]}"
        assert y.shape[1] == 1, f"Each output should contain 1 component. Got: {y.shape[1]}"
        assert len(labels) == len(np.unique(y)), f"Number of labels must match the number of unique y values. (labels: {len(labels)}, Unique y: {len(np.unique(y))})"

        N, N_f = X.shape

        self.N_l = len(labels)

        self.a = np.zeros((N_f, self.N_l ))
        self.b = np.zeros((self.N_l, ))

        y_flat = y.flatten()

        for j in range(self.N_l):
            N_j, _ = X[y_flat == j].shape
            self.b[j] = N_j/N                                                   # Class Priors -> N_j/N
            for i in range(N_f):
                N_ij, _ = X[np.logical_and(X[:, i] == 1, y_flat == j)].shape    # Partitions the np.ndarray where the feature is used and is part of a certain label
                self.a[i][j] = N_ij/N_j if N_ij != 0 else (N_ij + 1)/(N_j + 2)  # Likelihood -> N_ij/N_j

    def __alpha(self, j: int, val: np.ndarray) -> float:
        """
        Calculates the alpha value for the jth class label with specific val

        Args:
            j (int): jth class label
            val (np.ndarray (Shape: (9635, 1))): An array consisting of the usage of 9635 terms for the article

        Returns:
            float: alpha value for the jth class label with specific val
        """

        assert val.shape == (9635, ), f"val must be the correct shape of (9635, ). Got: {val.shape}"

        return np.sum(np.log(self.a[:, j][val == 1])) + np.sum(np.log(1 - self.a[:, j][val == 0])) + self.b[j]

    def __predict_label(self, val: np.ndarray) -> int:
        """
        Finds the best label for a specific value in X

        Args:
            val (np.ndarray (Shape: (9635, 1))): An array consisting of the usage of 9635 terms for the article

        Returns:
            int: Best Labels for the value
        """

        assert val.shape == (9635, ), f"val must be the correct shape of (9635, ). Got: {val.shape}"

        alphas = np.array([self.__alpha(l, val) for l in range(self.N_l)])
        gamma = np.min(alphas)

        return np.argmax(np.exp(alphas - gamma))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the best label for new X

        Args:
            X  (np.ndarray (Shape: (N, 9635))): A array consisting of each article with the usage of 9635 terms

        Returns:
            np.ndarray (Shape: (N, 1)): A array consisting of the label index for the related article
        """

        assert X.shape[1] == 9635, f"Each input should contain two components. Got: {X.shape[1]}"

        predictions = np.apply_along_axis(self.__predict_label, 1, X)
        return predictions.reshape((predictions.shape[0], 1))