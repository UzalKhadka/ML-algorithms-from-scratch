import numpy as np
import pandas as pd

# shuffle datasets
def _shuffle_train_test(X, y, random_state):
    if random_state is not None:
        np.random.seed(random_state)

    X_len = len(X)
    random_indices = np.arange(X_len)
    np.random.shuffle(random_indices)

    X_copy = X.copy()
    y_copy = y.copy()

    for i in range(X_len):
        rand_index = random_indices[i]

        # a simple hack to make X_copy and y_copy indexable
        if type(X_copy) == pd.DataFrame:
            X_copy.iloc[i] = X_copy.iloc[rand_index]
            y_copy.iloc[i] = y_copy.iloc[rand_index]

        else:  # supposing X_copy is a numpy array or python list or pd.Series
            X_copy[i] = X[rand_index]
            y_copy[i] = y[rand_index]

    return X_copy, y_copy


# get train and test data lengths
def _get_train_test_len(X_len, train_size=None, test_size=None):
    # if train_size and test_size are not given, then split the dataset into 75% train and 25% test
    if train_size is None and test_size is None:
        train_size = 0.75
        test_size = 0.25

    else:
        # if train size is given, then calculate test size
        if train_size is not None:
            test_size = 1 - train_size

        # if test size is given, then calculate train size
        if test_size is not None:
            train_size = 1 - test_size

        # check if train_size is given and if it is valid
        assert (
            train_size is not None and train_size > 0 and train_size < 1
        ), "train_size must be between 0 and 1"

        # check if train_size is given and if it is valid
        assert (
            test_size is not None and test_size > 0 and test_size < 1
        ), "test_size must be between 0 and 1"

        # check if train_size and test_size are given and they sum upto 1
        assert (
            train_size is not None
            and test_size is not None
            and train_size + test_size == 1
        ), "train_size and test_size must sum upto 1"

    # gettting train and test lengths
    train_len = int(X_len * train_size)
    test_len = X_len - train_len

    return train_len, test_len


# train_test_split function
def train_test_split(
    X, y, train_size=None, test_size=None, random_state=None, shuffle=True
):
    X_len = len(X)
    assert X_len == len(y), "X and y must have same length"

    # shuffle datasets if shuffle is True (default)
    if shuffle:
        X, y = _shuffle_train_test(X, y, random_state)

    # get train and test data lengths
    train_len, test_len = _get_train_test_len(X_len, train_size, test_size)

    # slicing X and y into train and test set
    X_train = X[:train_len]
    X_test = X[train_len:]
    y_train = y[:train_len]
    y_test = y[train_len:]

    return X_train, X_test, y_train, y_test
