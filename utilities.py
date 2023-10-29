import random
import numpy as np
from implementations import *


def unison_shuffled_copies(a, b):

    """
    Shuffle two arrays in unison, preserving their correspondence.

    This function ensures that the shuffling of both arrays keeps them in sync, i.e., 
    the i-th element in array 'a' will still correspond to the i-th element in array 'b' 
    after the shuffling.

    Parameters:
    - a (np.ndarray): First array to be shuffled.
    - b (np.ndarray): Second array to be shuffled.

    Returns:
    - tuple: A tuple containing two arrays (a, b) after shuffling.

    Raises:
    - AssertionError: If the input arrays 'a' and 'b' have different lengths.
    """
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing. If ratio times the number of samples is not round
    you can use np.floor. Also check the documentation for np.random.permutation,
    it could be useful.

    Args:
        x: numpy array of shape (N,), N is the number of samples.
        y: numpy array of shape (N,).
        ratio: scalar in [0,1]
        seed: integer.

    Returns:
        x_tr: numpy array containing the train data.
        x_te: numpy array containing the test data.
        y_tr: numpy array containing the train labels.
        y_te: numpy array containing the test labels.

    >>> split_data(np.arange(13), np.arange(13), 0.8, 1)
    (array([ 2,  3,  4, 10,  1,  6,  0,  7, 12,  9]), array([ 8, 11,  5]), array([ 2,  3,  4, 10,  1,  6,  0,  7, 12,  9]), array([ 8, 11,  5]))
    """
    # set seed
    np.random.seed(seed)

    x_shuffle, y_shuffle = unison_shuffled_copies(x, y)
    split_pos = round(len(x_shuffle) * ratio)

    return (
        x_shuffle[:split_pos],
        x_shuffle[split_pos:],
        y_shuffle[:split_pos],
        y_shuffle[split_pos:],
    )


def predict_logistic(X, w, threshold):
    """ BInary prediction on data

    Args:
        X np.array (D,): samples on which we want to predict
        w np.array (N,D): weights of the logistic model
        threshold int : threshold to change 

    Returns:
        _type_: _description_
    """
    return (sigmoid(X @ w) >= threshold).flatten()


def normalize(data):
    """
    Normalize the input data to have mean 0 and standard deviation 1.

    Parameters:
    - data: numpy array of shape (m, n) where m is the number of samples and n is the number of features.

    Returns:
    - normalized_data: numpy array of shape (m, n) with normalized values.
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    normalized_data = (data - mean) / (std + 10e-300)

    return normalized_data


def k_fold_cross_validation(X, y, model, k, model_params, threshold=0.5):
    """
    Perform k-fold cross-validation.

    Parameters:
    - X: features, numpy array of shape (num_samples, num_features)
    - y: targets, numpy array of shape (num_samples, )
    - model: a classifier having fit and predict methods
    - k: number of folds
    - model_params: dictionary with values of model paramters

    Returns:
    - mean_accuracy: the average accuracy over the k-folds
    """
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    fold_size = num_samples // k
    accuracies = []
    f1_scores = []

    for i in range(k):
        # Split data into train and test for this fold
        test_indices = indices[i * fold_size : (i + 1) * fold_size]
        train_indices = np.setdiff1d(indices, test_indices)

        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        # Fit model and predict
        w, loss = model(y_train, X_train, **model_params)

        y_pred = predict_logistic(X_test, w, threshold)

        # Calculate accuracy for this fold and append to accuracies list
        accuracy = np.mean(y_pred == y_test)
        f1 = compute_f1(y_test, y_pred)
        accuracies.append(accuracy)
        f1_scores.append(f1)

    # Calculate mean accuracy over all k-folds
    mean_accuracy = np.mean(accuracies)
    f1_score = np.mean(f1_scores)

    return mean_accuracy, f1_score


def hyperparameter_tuning(X, y, model, lambdas, gammas, model_params, k=5):
    """
    Tune hyperparameter using k-fold cross-validation.

    Parameters:
    - X: features
    - y: targets
    - model_class: a class of the model that accepts the hyperparameter in its constructor
    - param_name: name of the hyperparameter to be tuned
    - param_values: list of values for the hyperparameter
    - k: number of folds for cross-validation

    Returns:
    - best_param_value: the value of the hyperparameter that gives the best cross-validation accuracy
    """
    best_accuracy = 0
    best_f1_score = 0
    best_param_lambda = None
    best_param_gamma = None

    for gamma in gammas:
        for lambda_ in lambdas:
            model_params["lambda_"] = lambda_
            model_params["gamma"] = gamma
            accuracy, f1_score = k_fold_cross_validation(X, y, model, k, model_params)

            if f1_score >= best_f1_score and accuracy >= best_accuracy:
                best_accuracy = accuracy
                best_f1_score = f1_score
                best_param_lambda = lambda_
                best_param_gamma = gamma

            print(
                f" lambda= {lambda_}, gamma= {gamma}, CV accuracy = {accuracy:.4f}, f1_score = {f1_score:.4f}"
            )

    return best_param_lambda, best_param_gamma


def compute_f1(y_true, y_pred):
    """
    Compute the F1 score based on the true and predicted labels.

    Parameters:
    - y_true (numpy.ndarray): True labels array of shape (n_samples, ).
    - y_pred (numpy.ndarray): Predicted labels array of shape (n_samples, ).

    Returns:
    - float: F1 score for the given true and predicted labels.
    
    Raises:
    - AssertionError: If the length of y_true and y_pred arrays do not match.
    """

    assert (
        y_true.shape[0] == y_pred.shape[0]
    ), "Mismatched length between y_true and y_pred."

    # True Positives
    TP = np.sum(np.logical_and(y_pred == 1, y_true == 1))

    # False Positives
    FP = np.sum(np.logical_and(y_pred == 1, y_true == 0))

    # False Negatives
    FN = np.sum(np.logical_and(y_pred == 0, y_true == 1))

    # Precision and Recall
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0

    # F1 Score
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) != 0
        else 0
    )

    return f1


def clean_X_0(data):
    """
    Clean the given data by removing the first column, setting NaN values to 0, 
    and adding a column of ones as the first column after normalization.

    Parameters:
    - data (numpy.ndarray): Input data of shape (n_samples, n_features).

    Returns:
    - numpy.ndarray: Cleaned and normalized data with a column of ones prepended.
    """
    data_copied = np.array(data)
    data_copied = data_copied[:, 1:]  # remove ids
    data_copied[np.isnan(data_copied)] = 0
    data_copied = normalize(data_copied)
    ones_column = np.ones((data_copied.shape[0], 1)) # add columsn of ones
    return np.hstack((ones_column, data_copied))


def clean_X_mean(data):
    """
    Clean the given data by removing the first column, imputing NaN values with 
    the mean of their respective columns, and adding a column of ones 
    as the first column after normalization.

    Parameters:
    - data (numpy.ndarray): Input data of shape (n_samples, n_features).

    Returns:
    - numpy.ndarray: Cleaned and normalized data with a column of ones prepended.
    """
    data_copied = np.array(data)
    data_copied = data_copied[:, 1:]  # remove ids
    column_means = np.nanmean(data_copied, axis=0)
    data_copied[np.isnan(data_copied)] = np.take(
        column_means, np.where(np.isnan(data_copied))[1]
    )
    data_copied = normalize(data_copied)
    ones_column = np.ones((data_copied.shape[0], 1)) # add columsn of ones
    return np.hstack((ones_column, data_copied))


def clean_X_median(data):
    """
    Clean the given data by removing the first column, imputing NaN values with 
    the median of their respective columns, and normalizing the resulting data.

    Parameters:
    - data (numpy.ndarray): Input data of shape (n_samples, n_features).

    Returns:
    - numpy.ndarray: Cleaned and normalized data.
    """
    data_copied = np.array(data)
    data_copied = data_copied[:, 1:]   # remove ids
    column_medians = np.nanmedian(data_copied, axis=0)
    data_copied[np.isnan(data_copied)] = np.take(
        column_medians, np.where(np.isnan(data_copied))[1]
    )
    data_copied = normalize(data_copied)
    return data_copied


def clean_Y(y_data):
    """
    Clean the given target data by removing the first column (ids) and converting 
    any value of -1 to 0.

    Parameters:
    - y_data (numpy.ndarray): Target data of shape (n_samples, 1).

    Returns:
    - numpy.ndarray: Cleaned target data.
    """
    y_data_copied = np.array(y_data)
    y_data_copied = y_data_copied[:, 1]  # remove ids
    y_data_copied[y_data_copied == -1] = 0  # set -1 to 0
    return y_data_copied


def columns_to_remove(x_data, t):
    """
    Identify columns in the data that have an excessive number of specified outlier values.

    Parameters:
    - x_data (np.ndarray): Input data matrix where rows are samples and columns are features.
    - t (float): Threshold for the percentage of outlier values beyond which a column should be considered for removal.

    Returns:
    - np.ndarray: Array of column indices that should be removed.

    Notes:
    - This function checks for the presence of NaN values and specific outlier values (77, 99, 777, 999, 7777, 9999, 777777, 999999)
      in each column of the data. If the combined percentage of these values in a column exceeds the threshold 't', the 
      column index is added to the removal list.
    """
    remove = np.array([])

    for j in range(x_data.shape[1]):
        # focus: columns with too many nan values
        num_nan = np.sum(np.isnan(x_data[:, j]))
        per_nan = num_nan / x_data.shape[0]

        # focus: columns with too many 77/99 values
        num_2729 = np.sum(x_data[:, j] == 77) + np.sum(x_data[:, j] == 99)
        per_2729 = num_2729 / x_data.shape[0]

        # focus: columns with too many 77/99 values
        num_3739 = np.sum(x_data[:, j] == 777) + np.sum(x_data[:, j] == 999)
        per_3739 = num_3739 / x_data.shape[0]

        # focus: columns with too many 7777/9999 values
        num_4749 = np.sum(x_data[:, j] == 7777) + np.sum(x_data[:, j] == 9999)
        per_4749 = num_4749 / x_data.shape[0]

        # focus: columns with too many 777777/999999 values
        num_6769 = np.sum(x_data[:, j] == 777777) + np.sum(x_data[:, j] == 999999)
        per_6769 = num_6769 / x_data.shape[0]

        # find out which features to remove
        num_remove = num_nan + num_2729 + num_3739 + num_4749 + num_6769
        per_remove = num_remove / x_data.shape[0]
        if per_remove >= t:
            remove = np.append(remove, j)

    return remove.astype(int)


def reduced_data(x_data, t=0.6):
    """
    Return a version of the input data with columns removed based on the criteria defined in `columns_to_remove`.

    Parameters:
    - x_data (np.ndarray): Input data matrix where rows are samples and columns are features.
    - t (float, optional): Threshold for the percentage of outlier values beyond which a column should be removed.
                            Default is 0.6.

    Returns:
    - np.ndarray: Filtered data with specified columns removed.
    """
    filtered_data = np.delete(x_data, columns_to_remove(x_data, t), 1)
    return filtered_data



def calculate_metrics(y_true, y_pred):
    """
    Calculate TP, FP, TN, and FN for binary classification.

    Parameters:
    - y_true: list of true labels (0 or 1)
    - y_pred: list of predicted labels (0 or 1)

    Returns:
    - TP, FP, TN, FN
    """

    TP = FP = TN = FN = 0

    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            TP += 1
        elif yt == 0 and yp == 1:
            FP += 1
        elif yt == 1 and yp == 0:
            FN += 1
        elif yt == 0 and yp == 0:
            TN += 1

    return TP, FP, TN, FN


def drop_highly_correlated_features(data, threshold=0.95):
    """
    Drops columns in a numpy array that are highly correlated with others.

    Parameters:
    - data: A numpy array.
    - threshold: Correlation threshold above which columns are dropped.

    Returns:
    - A numpy array with highly correlated columns removed.
    """

    # Calculate the correlation matrix
    corr_matrix = np.corrcoef(data, rowvar=False)

    # Create a mask of size equal to the number of features in data
    drop_cols = np.zeros(corr_matrix.shape[0], dtype=bool)

    # For each feature, check if it's correlated with another feature more than the threshold
    for i in range(corr_matrix.shape[0]):
        for j in range(i + 1, corr_matrix.shape[0]):
            if abs(corr_matrix[i, j]) > threshold:
                drop_cols[j] = True

    # Return the data with the columns to drop removed
    return data[:, ~drop_cols], drop_cols

def drop_test_correlated_features(x_test, drop_cols): 
    """
    Drops same set of columns which were dropped in the training set on the test set. 

    Parameters:
    - x_test: A numpy array (the test set)
    - drop_cols: list of columns to be dropped 

    Returns:
    - test set without drop_cols
    """

    return x_test[:, ~drop_cols]

