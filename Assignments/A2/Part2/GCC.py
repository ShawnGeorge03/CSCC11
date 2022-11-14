__author__ = ["Shawn Santhoshgeorge (1006094673)", "Anaqi Amir Razif (1005813880)"]

from helper import *

class GCC:
    """
    Creates a Gaussian Class Conditionals Classifier
    """

    def __init__(self) -> None:
        """
        Initializes a Gaussian Class Conditionals Classifier
        """

        self.means = np.ndarray         # Means
        self.vars = np.ndarray          # Variences
        self.log_prob_l = np.ndarray    # Log of Class Priors

    def train(self, X: np.ndarray, y: np.ndarray, labels: list[str], var_bias: float = 1e-9) -> None:
        """
        Calculates the Means, Variance and Log Class Priors for each Gaussians

        Args:
            X  (np.ndarray (Shape: (N, 9635))): A array consisting of each article with the usage of 9635 terms
            y  (np.ndarray (Shape: (N, 1))): A array consisting of the label index for the related article
            label (str[]): Classes for each unique y
            var_bias (float, optional): Since each X_i is independent then the many of locations of the
                                            matrix will be empty and to ensure no DivisonZero Error,
                                            we add a small value to fill the zeros. The default choosen
                                            by trial and error to ensure the testing accuracy was the highest.
                                            Defaults to 1e-9
        """

        assert X.shape[0] == y.shape[0], f"Number of inputs and outputs are different. (X: {X.shape[0]}, y: {y.shape[0]})"
        assert X.shape[1] == 9635, f"Each input should contain two components. Got: {X.shape[1]}"
        assert y.shape[1] == 1, f"Each output should contain 1 component. Got: {y.shape[1]}"
        assert len(labels) == len(np.unique(y)), f"Number of labels must match the number of unique y values. (labels: {len(labels)}, Unique y: {len(np.unique(y))})"

        N, N_f = X.shape

        self.N_l = len(labels)

        self.means = np.zeros((self.N_l, N_f))
        self.vars = np.zeros((self.N_l, N_f))
        self.log_prob_l = np.zeros((self.N_l, ))

        y_flat = y.flatten()

        for l in range(self.N_l):
            X_l = X[y_flat == l]
            N_j, _ = X_l.shape
            self.means[l] = np.mean(X_l, axis=0)          # Means
            self.vars[l] = np.var(X_l, axis=0) + var_bias # Variances
            self.log_prob_l[l] = np.log(N_j) - np.log(N)  # Log of Class Priors

    def __predict_label(self, val: np.ndarray) -> int:
        """
        Finds the best label for a specific value in X

        Args:
            val (np.ndarray (Shape: (9635, 1))): An array consisting of the usage of 9635 terms for the article

        Returns:
            int: Best Labels for the value
        """

        assert val.shape == (9635, ), f"val must be the correct shape of (9635, ). Got: {val.shape}"

        preds = np.zeros((self.N_l, ))

        for l in range(self.N_l):
            # Calculates using the log of the Normal Distribution
            dist = np.power((val - self.means[l]), 2)/self.vars[l]
            preds[l] = -0.5 * np.sum(np.log(2 * np.pi * self.vars[l]) + dist) + self.log_prob_l[l]

        return np.argmax(preds)

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