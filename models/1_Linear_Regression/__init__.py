import numpy as np
import pandas as pd

# Simple Linear Regression Model using Least Squares Method
class LinearRegression:
    def __init__(self, fit_intercept=True, copy_X=True, n_jobs=None):
        self.coef_ = None  # coefficients i.e. m1, m2, m3, ... mn
        self.intercept_ = None  # intercept i.e. c
        self.n_features_in_ = None  # number of features
        self.feature_names_in_ = None  # names of features if available
        self.res_ = None  # sum of squared errors i.e. residuals

    # get the feature names if available (accepts DataFrame, Series, List, Numpy Array) else returns None
    def _get_feature_names(self, X):
        if type(X) == pd.DataFrame:
            return X.columns
        elif type(X) == pd.Series:
            return X.name
        elif type(X) == np.ndarray:
            return np.arange(X.shape[1])
        elif type(X) == list:
            return np.arange(len(X))
        else:
            return None

    def fit(self, X, y):
        # getting the number of features and the feature names
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = self._get_feature_names(X)

        # we use np.linalg.lstsq to find the coefficients and the y-intercept
        X = np.array(X)  # converting to numpy array
        X = np.append(
            X, np.ones(len(X), dtype=int).reshape(-1, 1), axis=1
        )  # adding an extra row of ones

        # finding the coefficients and the y-intercept
        coef = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        self.coef_ = coef[:-1]
        self.intercept_ = coef[-1]
        self.res_ = np.sum((y - X.dot(coef)) ** 2)

        return self

    # predicting the values Y = m1X1 + m2X2 + m3X3 + ... + mnXn + c
    def predict(self, X):
        X = np.array(X)
        X = np.append(
            X, np.ones(len(X), dtype=int).reshape(-1, 1), axis=1
        )  # adding an extra row of ones to get the y-intercept

        return X.dot(np.append(self.coef_, self.intercept_))

    # R2 score # coefficient of determination # r2 = 1 - (residuals / total sum of squares)
    def score(self, X, y):
        # predicting the values first
        y_pred = self.predict(X)

        # calculating the residuals and total sum of squares
        SSres = np.sum((y - y_pred) ** 2)
        SStot = np.sum((y - np.mean(y)) ** 2)

        # applying the formula
        r2_score = 1 - SSres / SStot

        return r2_score

    # get parameters
    def get_params(self):
        return {
            "coef_": self.coef_,
            "intercept_": self.intercept_,
            "n_features_in_": self.n_features_in_,
            "feature_names_in_": self.feature_names_in_,
            "res_": self.res_,
        }

    # set parameters
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)

        return self
