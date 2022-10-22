import numpy as np

# accuracy score function
# accuracy = (tp + tn) / (tp + tn + fp + fn)
# accuracy = (tp + tn) / total
def accuracy_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return np.sum(y_true == y_pred) / len(y_true)
