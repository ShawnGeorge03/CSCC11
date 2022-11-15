__author__ = ["Shawn Santhoshgeorge (1006094673)", "Anaqi Amir Razif (1005813880)"]

from helper import *

import numpy as np

class KNN:
    """
    Creates a k-Nearest Neighbors Classifier
    """

    def __init__(self, k: int) -> None:
        """
        Initializes a k-Nearest Neighbors Classifier

        Args:
            k (int): Number of a nearest neighbors used for classification
        """

        self.k = k

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Setups up the k-Nearest Neighbors Classifier

        Args:
            X  (np.ndarray (Shape: (N, 9635))): A array consisting of each article with the usage of 9635 terms
            y  (np.ndarray (Shape: (N, 1))): A array consisting of the label index for the related article
        """

        assert X.shape[0] == y.shape[0], f"Number of inputs and outputs are different. (X: {X.shape[0]}, y: {y.shape[0]})"
        assert X.shape[1] == 9635, f"Each input should contain two components. Got: {X.shape[1]}"
        assert y.shape[1] == 1, f"Each output should contain 1 component. Got: {y.shape[1]}"

        self.X_train = X
        self.y_train = y

    def __K_Nearest(self, val: np.ndarray) -> np.ndarray:
        """
        Find the k-Nearest Neighbors for a specific value in X

        Args:
            val (np.ndarray (Shape: (9635, 1))): An array consisting of the usage of 9635 terms for the article

        Returns:
            np.ndarray: Labels for the k-Nearest Neighbors of value
        """

        assert val.shape == (9635, ), f"val must be the correct shape of (9635, 1). Got: {val.shape}"

        distances = np.linalg.norm(self.X_train - val, axis=1)              # Calculates | x_i - c_i |
        nearest_neighbors = np.argpartition(distances, self.k)[:self.k]     # Finds the k-closest rows
        return self.y_train[nearest_neighbors].flatten()                    # Finds the labels

    def __predict_label(self, val: np.ndarray) -> int:
        """
        Finds the best label for a specific value in X

        Args:
            val (np.ndarray (Shape: (9635, 1))): An array consisting of the usage of 9635 terms for the article

        Returns:
            int: Best Labels for the value
        """
        assert val.shape == (9635, ), f"val must be the correct shape of (9635, 1). Got: {val.shape}"

        nearest_neighbors = self.__K_Nearest(val)                          # Gets the Nearest Neighbors
        labels, counts = np.unique(nearest_neighbors, return_counts=True)
        max_counts = counts == counts.max()
        if np.count_nonzero(max_counts) > 1:                               # If there are ties choose the label randomly
            tie_labels = labels[max_counts]
            return tie_labels[np.random.randint(len(tie_labels))]
        else:
            return labels[np.argmax(counts)]                               # Uses Index of Max Count to get label

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
