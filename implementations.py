import random
import numpy as np


def compute_mse(y, tx, w):
    """Calculate the loss using either MSE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """

    e = y - np.dot(tx, w)
    squared_error = np.square(e)
    mse = 0.5 * np.mean(squared_error)
    return mse


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD
    """

    weights = initial_w
    loss = compute_mse(y, tx, weights)

    for n_iter in range(max_iters):
        error = y - tx.dot(weights)
        gradient = -1 / len(tx) * np.dot(tx.T, error)  # compute gradient

        weights = weights - gamma * gradient  # update weights
        loss = compute_mse(y, tx, weights)

    return weights, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
    """

    weights = initial_w
    loss = compute_mse(y, tx, weights)

    for n_iter in range(max_iters):
        n = random.randint(0, len(tx) - 1)
        gradient = -tx[n] * (y[n] - tx[n] @ weights)  # compute gradient

        weights = weights - gamma * gradient  # update weights
        loss = compute_mse(y, tx, weights)

    return weights, loss


def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    """

    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    mse = compute_mse(y, tx, w)

    return w, mse


def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.

    """

    N, D = tx.shape
    w = np.linalg.solve(
        np.dot(tx.T, tx) + np.identity(D) * (lambda_ * (2 * N)), np.dot(tx.T, y)
    )
    mse = compute_mse(y, tx, w)

    return w, mse


def sigmoid(x):
    """Apply the logistic function.
    Args:
        x: numpy array of shape (N,), input to sigmoid function

    Returns:
        sigmoid function
    """
    return 1.0 / (1 + np.exp(-x))


def compute_logistic_loss(y, tx, weights):
    """
    Computes sigmoids and loss considering the current weights.
    y: np.array
        The target values
    tx: np.array
        The data matrix (each row is a data point)
    weights: np.array
        Current model weights
    """

    pred = tx @ weights
    sigmoids = 1.0 / (1 + np.exp(-pred))
    loss = -np.mean(y * np.log(sigmoids) + (1 - y) * np.log(1 - sigmoids))
    return sigmoids, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Perform logistic regression using gradient descent.

    Parameters:
    y: np.array
        The target values
    tx: np.array
        The data matrix (each row is a data point)
    initial_w: np.array
        Initial weights
    max_iters: int
        Maximum number of iterations for gradient descent
    gamma: float
        Learning rate

    Returns:
    w: np.array
        Optimized weights after training
    loss: float
        Final loss after training on the training set
    """

    w = initial_w

    sigmoids, loss = compute_logistic_loss(y, tx, w)

    for iter in range(max_iters):
        # compute the gradient

        gradient = tx.T.dot(sigmoids - y) / len(tx)

        # update w through the negative gradient direction
        w = w - gamma * gradient

        sigmoids, loss = compute_logistic_loss(y, tx, w)

    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Perform regularized logistic regression using gradient descent.

    Parameters:
    y: np.array
        The target values
    tx: np.array
        The data matrix (each row is a data point)
    lambda_: float
        Regularization parameter
    initial_w: np.array
        Initial weights
    max_iters: int
        Maximum number of iterations for gradient descent
    gamma: float
        Learning rate

    Returns:
    w: np.array
        Optimized weights after training
    loss: float
        Final loss after training on the training set
    """

    w = initial_w
    sigmoids, loss = compute_logistic_loss(
        y, tx, w
    )  # compute sigmoids and loss for 0 iteration

    for _ in range(max_iters):
        gradient = tx.T.dot(sigmoids - y) / len(y) + 2 * lambda_ * w  # compute gradient
        w = w - gamma * gradient  # update weights
        sigmoids, loss = compute_logistic_loss(y, tx, w)

    return w, loss
