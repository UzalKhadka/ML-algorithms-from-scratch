import numpy as np

# Confusion Matrix Function
def confusion_matrix(y_true, y_pred, labels=None):
    assert len(y_true) == len(y_pred), "Length of y_true and y_pred must be equal"

    if labels is None or len(labels) == 0:
        labels = np.unique(
            y_true + y_pred
        )  # get all unique labels from both y_true and y_pred
    else:
        labels = np.unique(labels)

    len_labels = len(labels)  # number of unique labels
    # create a confusion matrix
    conf_matrix = np.zeros((len_labels, len_labels), dtype=int)

    # fill the confusion matrix
    for i in range(len(y_true)):  # for each y_true sample
        for j in range(
            len_labels
        ):  # iterating over all labels to find the index of y_true sample in labels
            if y_true[i] == labels[j]:  # check for match
                for k in range(
                    len_labels
                ):  # iterating over all labels to find the index of y_pred sample in labels
                    if (
                        y_pred[i] == labels[k]
                    ):  # check for match and now for y_pred sample
                        conf_matrix[j][
                            k
                        ] += 1  # increment the value of confusion matrix at the index of y_true and y_pred sample
                        break  # for optimization
                break  # for optimization

    return conf_matrix
