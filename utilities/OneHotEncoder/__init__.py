import numpy as np

# One Hot Encoding class
class OneHotEncoder:
    # this class can map only a single column at a time
    def __init__(self, sparse=True, dtype=float, handle_unknown="ignore"):
        self.sparse = sparse  # the sparse will always be true for now and the we will return a sparse matrix everytime
        self.dtype = dtype  # dtype -> float | int
        self.handle_unknown = handle_unknown  # handle_unknown will be 'ignore' for now and return None for unknown categorical values

        self.categories_ = None

    # fit the class with the incoming data
    def fit(self, X):
        # the input X is expected to be a single column value / series
        self.categories_ = np.unique(X)
        return self

    # transform the incoming data to one hot encoded form
    def transform(self, X):
        X = np.array(X)
        X_transformed = np.zeros((X.shape[0], len(self.categories_)), dtype=self.dtype)
        for i, category in enumerate(self.categories_):
            X_transformed[:, i] = (X == category).astype(int)
        return X_transformed

    # fitting the class with the incoming data and transforming it into one-hot encoded form
    def fit_transform(self, X):
        return self.fit(X).transform(X)

    # revering the encoded data into original form
    def inverse_transform(self, X):
        X = np.array(X)
        X_transformed = np.zeros(X.shape[0], dtype=self.categories_.dtype)

        for i, category in enumerate(self.categories_):
            X_transformed[X[:, i] == 1] = category

        # filtering the unknown categorical values (if any) to None instead of 0 since 0 is the default one if the category is not found
        return [x if x != 0 else None for x in X_transformed]
