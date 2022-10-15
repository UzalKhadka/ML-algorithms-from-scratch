import numpy as np

# R2 Score function
def r2_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # calculating the residuals and total sum of squares
    SSres = np.sum((y_true - y_pred) ** 2)
    SStot = np.sum((y_true - np.mean(y_true)) ** 2)

    # applying the formula
    r2_score = 1 - np.divide(
        SSres, SStot, out=np.zeros_like(SSres, dtype=type(SStot)), where=SStot != 0
    )  # using np.divide to avoid ZeroDivisionError

    return r2_score
