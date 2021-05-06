import numpy as np

def binary_classification_metrics(prediction: np.ndarray,
                                  ground_truth: np.ndarray,
                                  negative: int=0,
                                  positive: int=1) -> tuple:
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels
    negative: a sentinel value indicating a negative label
    positive: a sentinel value indicating a positive label

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''

    if negative or positive not in np.unique(
        np.append(np.unique(prediction), np.unique(ground_truth))
    ):
        raise Exception(
            'Default negative or positive are not conteined in input arrays. '
            'Specify these values')

    # accumulate the true/false negative/positives
    tp = np.sum(np.logical_and(prediction == positive, ground_truth == positive))
    tn = np.sum(np.logical_and(prediction == negative, ground_truth == negative))
    fp = np.sum(np.logical_and(prediction == positive, ground_truth == negative))
    fn = np.sum(np.logical_and(prediction == negative, ground_truth == positive))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = multiclass_accuracy(prediction, ground_truth)

    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''

    return np.mean(prediction == ground_truth)
