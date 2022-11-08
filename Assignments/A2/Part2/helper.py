import numpy as np
from sklearn.model_selection import train_test_split

from enum import Enum

class Mode(Enum):
    FREQ = 'Frequency'
    BINARY = 'Binary'

def load_data(mode: Mode, train_size: float, test_size: float, random_state: int =2):
    """
    Loads Data From BBC files

    Args:
        mode (Mode): Frequency or Binary values
        train_size (float): Ratio of the Data to be used for training
        test_size (float): Ratio of the Data to be used for testing
        random_state (int, optional): Controls the shuffling applied. Defaults to 2.

    Returns:
        tuple: Split Data into Train and Test, Terms Used, and Labels
    """

    assert train_size + test_size == 1, 'Train and Test Size must be 1'

    BBC_PATH = 'Assignments/A2/Part2/data/bbc'

    with open(f'{BBC_PATH}.terms', 'r') as terms_file:
        terms = {idx: term.strip() for idx, term in enumerate(terms_file)}

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

    return X_train, X_test, y_train, y_test, terms, labels