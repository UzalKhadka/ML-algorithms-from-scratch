# importing required libraries
import numpy as np
from warnings import warn

# precision score function / Sensitivity / Positive Predictive Value
# precision = tp / (tp + fp)
def precision_score(y_true, y_pred, average="binary", zero_division="warn"):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # for binary classification, we can use the formula straight away
    if average == "binary":
        assert (
            len(np.unique(y_true)) == 2
        ), "y_true should have only 2 unique values i.e. binary classification"

        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))

        if (
            tp + fp == 0
        ):  # if there are no true positives and false positives, we will get a division by zero error
            if zero_division == "warn":
                warn(
                    "set zero_division to 0 or 1 to avoid this warning. Default is 'warn'"
                )
                return 0  # since x / 0
            elif zero_division == 0:
                return 0
            elif zero_division == 1:
                return 1
            else:
                warn("RuntimeWarning: zero_division must be either 0, 1 or 'warn'")
                return 0  # since x / 0
        else:
            return tp / (tp + fp)

    # for multiclass classification, we need to calculate precision for each class
    elif average == "micro":
        tp = np.sum((y_true == y_pred))
        fp = np.sum((y_true != y_pred))

        if (
            tp + fp == 0
        ):  # if there are no true positives and false positives, we will get a division by zero error
            if zero_division == "warn":
                warn(
                    "set zero_division to 0 or 1 to avoid this warning. Default is 'warn'"
                )
                return 0  # since x / 0
            elif zero_division == 0:
                return 0
            elif zero_division == 1:
                return 1
            else:
                warn("RuntimeWarning: zero_division must be either 0, 1 or 'warn'")
                return 0  # since x / 0
        else:
            return tp / (tp + fp)

    # for multiclass classification, we need to calculate precision for each class and return the mean
    elif average == "macro":
        classes = np.unique(y_true)
        pre_scores = []

        for c in classes:
            tp = np.sum((y_true == c) & (y_pred == c))
            fp = np.sum((y_true != c) & (y_pred == c))

            if (
                tp + fp == 0
            ):  # if there are no true positives and false positives, we will get a division by zero error
                if zero_division == "warn":
                    warn(
                        "set zero_division to 0 or 1 to avoid this warning. Default is 'warn'"
                    )
                    pre_scores.append(0)  # since x / 0
                elif zero_division == 0:
                    pre_scores.append(0)
                elif zero_division == 1:
                    pre_scores.append(1)
                else:
                    warn("RuntimeWarning: zero_division must be either 0, 1 or 'warn'")
                    pre_scores.append(0)  # since x / 0
            else:
                pre_scores.append(tp / (tp + fp))

        return np.mean(pre_scores)

    # for multiclass classification, we need to calculate precision for each class and return the score for each class
    elif average == None:
        classes = np.unique(y_true)
        pre_scores = []

        for c in classes:
            tp = np.sum((y_true == c) & (y_pred == c))
            fp = np.sum((y_true != c) & (y_pred == c))

            if (
                tp + fp == 0
            ):  # if there are no true positives and false positives, we will get a division by zero error
                if zero_division == "warn":
                    warn(
                        "set zero_division to 0 or 1 to avoid this warning. Default is 'warn'"
                    )
                    pre_scores.append(0)  # since x / 0
                elif zero_division == 0:
                    pre_scores.append(0)
                elif zero_division == 1:
                    pre_scores.append(1)
                else:
                    warn("RuntimeWarning: zero_division must be either 0, 1 or 'warn'")
                    pre_scores.append(0)  # since x / 0
            else:
                pre_scores.append(tp / (tp + fp))

        return np.array(pre_scores)

    # Else, we raise an error
    else:
        raise ValueError(
            "average parameter must be 'binary', 'macro', 'micro', or None"
        )
