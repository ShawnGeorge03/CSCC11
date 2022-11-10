__author__ = ["Shawn Santhoshgeorge (1006094673)", "Anaqi Amir Razif (1005813880)"]

import enum

import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split


class Mode(enum.Enum):
    FREQ = 'Frequency'
    BINARY = 'Binary'

def load_data(mode: Mode, train_size: float, test_size: float, random_state: int = 2) -> tuple:
    """
    Loads Data From BBC files

    Args:
        mode (Mode): Frequency or Binary values
        train_size (float): Ratio of the Data to be used for training
        test_size (float): Ratio of the Data to be used for testing
        random_state (int, optional): Controls the shuffling applied. Defaults to 2.

    Returns:
        tuple: Train Subset, Test Subset, and Labels
    """

    assert train_size + test_size == 1, 'Train and Test Size must be 1'

    BBC_PATH = 'Assignments/A2/Part2/data/bbc'

    with open(f'{BBC_PATH}.mtx', 'r') as articles_file:
        articles_file.readline()
        rows, cols = map(int, articles_file.readline().split()[:2][::-1])
        X = np.empty((rows, cols), dtype=np.int8)
        for article in articles_file:
            col, row, freq = article.split()
            if mode.value == Mode.FREQ.value:
                X[int(row) - 1][int(col) - 1] = int(float(freq))
            elif mode.value == Mode.BINARY.value:
                X[int(row) - 1][int(col) - 1] = int(float(freq) > 1)

    with open(f'{BBC_PATH}.classes', 'r') as labels_file:
        labels_file.readline()
        labels_file.readline()
        labels = labels_file.readline().strip().split(" ")[-1].split(",")
        labels_file.readline()
        y = np.empty((rows, 1), dtype=np.uint8)
        for line in labels_file:
            row, label_idx = map(int, line.split())
            y[row] = label_idx

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, \
        test_size=test_size, random_state=random_state)

    return X_train, y_train, X_test, y_test, labels

def get_accuracy(labels: list[str], expected: np.ndarray, actual: np.ndarray) -> tuple:
    """
    Calculates the accuracy of each label and overall accuracy

    Args:

        labels (list[str]): Unique class labels
        expected (np.ndarray (Shape: (N, 1))): Expected Labels
        actual (np.ndarray (Shape: (N, 1))): Predicted Labels

    Returns:
        tuple: Label Accuracy, and Overall Accuracy
    """


    expected_flat = expected.flatten()
    actual_flat = actual.flatten()

    label_accuracy, total = [], 0

    for label_idx, label in enumerate(labels):
        expected_l = expected_flat[expected_flat == label_idx]
        actual_l = actual_flat[expected_flat == label_idx]
        count_l = np.count_nonzero(expected_l == actual_l)
        label_accuracy.append((label, f'{round(count_l/len(expected_l) * 100, 2)} %' ))
        total += count_l

    label_acc = DataFrame(label_accuracy, columns=['Label','Label Accuracy'])
    overall_acc = f'{round(total/len(expected_flat) * 100, 2)} %'

    return label_acc, overall_acc
