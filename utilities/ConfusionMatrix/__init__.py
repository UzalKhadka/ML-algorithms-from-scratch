import numpy as np

# Confusion Matrix Function
def confusion_matrix(y_true, y_pred, labels=None):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    assert len(y_true) == len(y_pred), "Length of y_true and y_pred must be equal"

    if labels is None or len(labels) == 0:
        # get all unique labels from both y_true and y_pred
        labels = np.unique(np.append(y_true, y_pred))
    else:
        labels = np.unique(labels)

    len_labels = len(labels)  # number of unique labels
    # create a confusion matrix
    conf_matrix = np.zeros((len_labels, len_labels), dtype=int)

    # fill the confusion matrix
    # for each y_true sample
    for i in range(len(y_true)):
        # iterating over all labels to find the index of y_true sample in labels
        for j in range(len_labels):
            # check for match
            if y_true[i] == labels[j]:
                # iterating over all labels to find the index of y_pred sample in labels
                for k in range(len_labels):
                    # check for match and now for y_pred sample
                    if y_pred[i] == labels[k]:
                        # increment the value of confusion matrix at the index of y_true and y_pred sample
                        conf_matrix[j][k] += 1
                        break  # for optimization
                break  # for optimization

    return conf_matrix
