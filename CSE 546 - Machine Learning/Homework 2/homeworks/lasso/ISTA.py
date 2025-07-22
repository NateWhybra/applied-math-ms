from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils import problem


@problem.tag("hw2-A")
def step(X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float, eta: float) -> Tuple[np.ndarray, float]:
    """S ingle step in ISTA algorithm.
    It should update every entry in weight, and then return an updated version of weight along with calculated bias on input weight!

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Weight returned from the step before.
        bias (float): Bias returned from the step before.
        _lambda (float): Regularization constant. Determines when weight is updated to 0, and when to other values.
        eta (float): Step-size. Determines how far the ISTA iteration moves for each step.

    Returns:
        Tuple[np.ndarray, float]: Tuple with 2 entries. First represents updated weight vector, second represents bias.
    
    """
    # raise NotImplementedError("Your Code Goes Here")

    # Update the bias.
    bias_new = bias - (2 * eta) * np.sum(X @ weight + bias - y)

    # Update the weights.
    weight_step = weight - (2 * eta) * (X.T @ (X @ weight + bias - y))
    weight_new = weight_step.copy()
    c = 2 * eta * _lambda
    weight_new[weight_step < -c] += c
    weight_new[np.abs(weight_step) <= c] = 0
    weight_new[weight_step > c] -= c

    # Return the updated weights and bias.
    return weight_new, bias_new


@problem.tag("hw2-A")
def loss(X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float) -> float:
    """L-1 (Lasso) regularized SSE loss.
    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Currently predicted weights.
        bias (float): Currently predicted bias.
        _lambda (float): Regularization constant. Should be used along with L1 norm of weight.

    Returns:
        float: value of the loss function
    """
    # raise NotImplementedError("Your Code Goes Here")

    # Compute and return the LASSO loss.
    return np.linalg.norm(X @ weight + (bias - y), 2) ** 2 + _lambda * np.linalg.norm(weight, 1)


@problem.tag("hw2-A", start_line=5)
def train(X: np.ndarray, y: np.ndarray, _lambda: float = 0.01, eta: float = 0.00001, convergence_delta: float = 1e-4, start_weight: np.ndarray = None, start_bias: float = None) -> Tuple[np.ndarray, float]:
    """Trains a model and returns predicted weight and bias.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        _lambda (float): Regularization constant. Should be used for both step and loss.
        eta (float): Step size.
        convergence_delta (float, optional): Defines when to stop training algorithm.
            The smaller the value the longer algorithm will train.
            Defaults to 1e-4.
        start_weight (np.ndarray, optional): Weight for hot-starting model.
            If None, defaults to array of zeros. Defaults to None.
            It can be useful when testing for multiple values of lambda.
        start_bias (float, optional): Bias for hot-starting model.
            If None, defaults to zero. Defaults to None.
            It can be useful when testing for multiple values of lambda.

    Returns:
        Tuple[np.ndarray, float]: A tuple with first item being array of shape (d,) representing predicted weights,
            and second item being a float representing the bias.

    Note:
        - You will have to keep an old copy of weights for convergence criterion function.
            Please use `np.copy(...)` function, since numpy might sometimes copy by reference,
            instead of by value leading to bugs.
        - You will also have to keep an old copy of bias for convergence criterion function.
        - You might wonder why do we also return bias here, if we don't need it for this problem.
            There are two reasons for it:
                - Model is fully specified only with bias and weight.
                    Otherwise you would not be able to make predictions.
                    Training function that does not return a fully usable model is just weird.
                - You will use bias in next problem.
    """

    # Make space for necessary data.
    if start_weight is None:
        start_weight = np.zeros(X.shape[1])
        start_bias = 0
    # old_w: Optional[np.ndarray] = None
    # old_b: float = None
    weight_new = None
    bias_new = None

    # raise NotImplementedError("Your Code Goes Here")

    # Set old_w and old_b to be the starting weights and bias.
    old_w = start_weight.copy()
    old_b = start_bias

    # While the model hasn't converged.
    converged = False
    while not converged:
        # Compute weight and bias update.
        weight_new, bias_new = step(X=X, y=y, weight=old_w, bias=old_b, _lambda=_lambda, eta=eta)

        # Check if the model converged.
        converged = convergence_criterion(weight=weight_new, old_w=old_w, bias=bias_new, old_b=old_b, convergence_delta=convergence_delta)

        # Reset old_w and old_b.
        old_w = weight_new.copy()
        old_b = bias_new

    # After the model converges, return the new weights and bias.
    return weight_new, bias_new


@problem.tag("hw2-A")
def convergence_criterion(weight: np.ndarray, old_w: np.ndarray, bias: float, old_b: float, convergence_delta: float) -> bool:
    """Function determining whether weight and bias has converged or not.
    It should calculate the maximum absolute change between weight and old_w vector, and compare it to convergence delta.
    It should also calculate the maximum absolute change between the bias and old_b, and compare it to convergence delta.

    Args:
        weight (np.ndarray): Weight from current iteration of gradient descent.
        old_w (np.ndarray): Weight from previous iteration of gradient descent.
        bias (float): Bias from current iteration of gradient descent.
        old_b (float): Bias from previous iteration of gradient descent.
        convergence_delta (float): Aggressiveness of the check.

    Returns:
        bool: False, if weight and bias has not converged yet. True otherwise.
    """
    # raise NotImplementedError("Your Code Goes Here")

    # Compute weight/bias changes.
    delta_weight = np.max(np.abs(weight - old_w))
    delta_bias = np.abs(bias - old_b)

    # Check convergence criteria.
    weight_converged = delta_weight <= convergence_delta
    bias_converged = delta_bias <= convergence_delta
    converged = np.logical_and(weight_converged, bias_converged)

    return converged


def synthetic_data(n=500, d=1000, k=100, sigma_noise=1, mu_X=0, sigma_X=1):
    # Generate weights.
    w = np.zeros(shape=d)
    w[0:k] = (np.arange(k) + 1) / k

    # Generate noise.
    noise = np.random.normal(loc=0, scale=sigma_noise, size=n)

    # Generate X data.
    X = np.random.normal(loc=mu_X, scale=sigma_X, size=(n, d))

    # Generate y data.
    y = X @ w + noise

    # Return X, y data.
    return X, y, w


# I'm pretty sure we need to use the standardized data here.
def get_lambda_range(X, y, c):
    # Calculate lambda_max.
    lambda_max = 2 * np.max(np.abs(X.T @ (y - np.mean(y))))

    # Decrease lambda_max by dividing by powers of 2.
    num_lamda = int(np.floor(np.abs(np.log2(lambda_max) - np.log2(c)))) + 2
    lambda_values = np.array([lambda_max / (2 ** i) for i in range(num_lamda)])

    # Return the lambda_values and the number of values.
    return lambda_values, num_lamda


@problem.tag("hw2-A")
def main(n=500, d=1000, k=100, sigma_noise=1, sigma_X=1, mu_X=5, eta=0.00001, convergence_delta=1e-4, lambda_thresh=0.1):
    """
    Use all of the functions above to make plots.
    """
    # raise NotImplementedError("Your Code Goes Here")

    # Get synthetic data.
    X, y, true_weights = synthetic_data(n=n, d=d, k=k, sigma_noise=sigma_noise, sigma_X=sigma_X, mu_X=mu_X)

    # Standardize X.
    X_standard = (X - np.mean(X, axis=0)) / (np.std(X, axis=0))

    # Get lambda range.
    lambda_values, num_lambda = get_lambda_range(X=X_standard, y=y, c=lambda_thresh)

    # Make storage to save the weights for each lambda.
    all_weights = np.zeros(shape=(d, num_lambda))
    all_bias = np.zeros(shape=num_lambda)

    # For each value of lambda, train a model.
    for i, _lambda in np.ndenumerate(lambda_values):
        model_weights, model_bias = train(X=X_standard, y=y, _lambda=_lambda, eta=eta, convergence_delta=convergence_delta, start_weight=np.zeros(d), start_bias=0)
        all_weights[:, i] = model_weights.reshape((d, 1))
        all_bias[i] = model_bias

    # For each value of lambda, count how many weights are not zero.
    non_zero_weights = np.sum(all_weights != 0, axis=0)

    # Compute FPR and TPR.
    incorrect_non_zero_counts = np.zeros(shape=num_lambda)
    correct_non_zero_counts = np.zeros(shape=num_lambda)
    for i in range(num_lambda):
        current_weights = all_weights[:, i].reshape(d)
        incorrect_non_zero_counts[i] = np.sum((current_weights != 0) & (true_weights == 0))
        correct_non_zero_counts[i] = np.sum((current_weights != 0) & (true_weights != 0))
    fdr = incorrect_non_zero_counts / (non_zero_weights + 1e-8)
    tpr = correct_non_zero_counts / k

    # Plot the number of non-zero weights vs lambda (Plot 1).
    fig, ax = plt.subplots()
    ax.scatter(lambda_values, non_zero_weights)
    plt.xscale('log')
    plt.xlabel(r"$\lambda$")
    plt.ylabel("Number of Non-Zero Weights")
    plt.title("Plot 1")
    plt.show()

    # Plot TPR vs. FDR (Plot 2).
    fig, ax = plt.subplots()
    ax.scatter(fdr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(r"Plot 2 ($\lambda$ decreases from left to right!)")
    plt.show()


if __name__ == "__main__":
    main()
