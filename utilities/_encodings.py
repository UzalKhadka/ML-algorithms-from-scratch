import numpy as np

# Label Encoding class
class LabelEncoder:
    # initialize the class
    def __init__(self):
        self.classes_ = None

    # fit the class with the incoming data
    def fit(self, X):
        self.classes_ = np.unique(X)
        return self

    # transform the incoming data into encoded form
    def transform(self, X):
        X = np.array(X)
        X_transformed = np.zeros(X.shape).astype(int)
        for i, val in enumerate(self.classes_):
            X_transformed[X == val] = i
        return X_transformed

    # fitting the class with the incoming data and transforming it into encoded form
    def fit_transform(self, X):
        return self.fit(X).transform(X)

    # revering the encoded data into original form
    def inverse_transform(self, X):
        X = np.array(X)
        X_transformed = np.zeros(X.shape).astype(self.classes_.dtype)
        for i, val in enumerate(self.classes_):
            X_transformed[X == i] = val
        return X_transformed

    # get the categorical classes
    def get_params(self, deep=True):
        return {"classes_": self.classes_}
