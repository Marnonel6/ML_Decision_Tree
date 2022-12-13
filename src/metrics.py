from cmath import isfinite
import numpy as np


def compute_confusion_matrix(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length
    numpy vector), compute the confusion matrix. The confusion
    matrix for a binary classifier would be a 2x2 matrix as follows:

    [
        [true_negatives, false_positives],
        [false_negatives, true_positives]
    ]

    You do not need to implement confusion matrices for labels with more
    classes. You can assume this will always be a 2x2 matrix.

    Compute and return the confusion matrix.

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        confusion_matrix (np.array): 2x2 confusion matrix between predicted and actual labels

    """

    # print('Actual = ', actual)
    # print('Predictions = ', predictions)
    # print('\n \n')

    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    

    for i in range(0,np.shape(actual)[0]):
        if predictions[i] == True and actual[i] == True:
            tp += 1
        elif predictions[i] == True and actual[i] == False:
            fp += 1
        elif predictions[i] == False and actual[i] == True:
            fn += 1
        elif predictions[i] == False and actual[i] == False:
            tn += 1

    confusion_matrix = np.array([[tn, fp],[fn,tp]])
    # print('confusion_matrix: ',  confusion_matrix)
    # print('\n \n')

    return confusion_matrix
    #raise NotImplementedError



def compute_accuracy(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length
    numpy vector), compute the accuracy:

    Hint: implement and use the compute_confusion_matrix function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        accuracy (float): accuracy
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    confusion_m = compute_confusion_matrix(actual,predictions)
    accuracy = (confusion_m[0,0] + confusion_m[1,1])/len(predictions)  # True positive and negative devide by total guesses

    # print('Actual = ', actual)
    # print('Predictions = ', predictions)
    # print('\n \n')
    # print('confusion_m = ', confusion_m)
    # print('accuracy = ', accuracy)
    # print('\n \n')

    return accuracy

    #raise NotImplementedError


def compute_precision_and_recall(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length
    numpy vector), compute the precision and recall:

    https://en.wikipedia.org/wiki/Precision_and_recall

    You MUST account for edge cases in which precision or recall are undefined
    by returning np.nan in place of the corresponding value.

    Hint: implement and use the compute_confusion_matrix function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output a tuple containing:
        precision (float): precision
        recall (float): recall
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    confusion_m = compute_confusion_matrix(actual,predictions)

    tp = confusion_m[1,1]
    fp = confusion_m[0,1]
    fn = confusion_m[1,0]

    p = tp/(tp+fp)
    r = tp/(tp+fn)

    # print('tp = ', tp)
    # print('fp = ', fp)
    # print('fn = ', fn)
    # print('\n \n')
    # print('p = ', p)
    # print('r = ', r)
    # print('\n \n')


    #ans = (p,r)

    if ~np.isfinite(p) and ~np.isfinite(r):
        #print('both nan')
        return (np.nan,np.nan)
    elif ~np.isfinite(p):
        #print('p nan')
        return (np.nan,r)
    elif ~np.isfinite(r):
        #print('r nan')
        return (p,np.nan)
    else: 
        #print('both good')
        return (p,r)


    

    #raise NotImplementedError


def compute_f1_measure(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length
    numpy vector), compute the F1-measure:

    https://en.wikipedia.org/wiki/Precision_and_recall#F-measure

    Because the F1-measure is computed from the precision and recall scores, you
    MUST handle undefined (NaN) precision or recall by returning np.nan. You
    should also consider the case in which precision and recall are both zero.

    Hint: implement and use the compute_precision_and_recall function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        f1_measure (float): F1 measure of dataset (harmonic mean of precision and
        recall)
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    pNr = compute_precision_and_recall(actual, predictions)
    p = pNr[0]
    r = pNr[1]

    if ~np.isfinite(p) or ~np.isfinite(r):
        #print('Both nan')
        return (np.nan)
    elif p == 0  and r == 0:
        #print('Both 0')
        return (np.nan)
    else: 
        F = 2*((p*r)/(p+r))
        #print('F = ', F)
        #print('\n \n')
        return F

    #raise NotImplementedError
