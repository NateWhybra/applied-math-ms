from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


def f_true(x: np.ndarray) -> np.ndarray:
    """True function, which was used to generate data.
    Should be used for plotting.

    Args:
        x (np.ndarray): A (n,) array. Input.

    Returns:
        np.ndarray: A (n,) array.
    """
    return 6 * np.sin(np.pi * x) * np.cos(4 * np.pi * x ** 2)


@problem.tag("hw3-A")
def poly_kernel(x_i: np.ndarray, x_j: np.ndarray, d: int) -> np.ndarray:
    """Polynomial kernel.

    Given two indices a and b it should calculate:
    K[a, b] = (x_i[a] * x_j[b] + 1)^d

    Args:
        x_i (np.ndarray): An (n,) array. Observations (Might be different from x_j).
        x_j (np.ndarray): An (m,) array. Observations (Might be different from x_i).
        d (int): Degree of polynomial.

    Returns:
        np.ndarray: A (n, m) matrix, where each element is as described above (see equation for K[a, b])

    Note:
        - It is crucial for this function to be vectorized, and not contain for-loops.
            It will be called a lot, and it has to be fast for reasonable run-time.
        - You might find .outer functions useful for this function.
            They apply an operation similar to xx^T (if x is a vector), but not necessarily with multiplication.
            To use it simply append .outer to function. For example: np.add.outer, np.divide.outer
    # """
    # raise NotImplementedError("Your Code Goes Here")
    return (np.outer(x_i, x_j) + 1) ** d


@problem.tag("hw3-A")
def rbf_kernel(x_i: np.ndarray, x_j: np.ndarray, gamma: float) -> np.ndarray:
    """Radial Basis Function (RBF) kernel.

    Given two indices a and b it should calculate:
    K[a, b] = exp(-gamma*(x_i[a] - x_j[b])^2)

    Args:
        x_i (np.ndarray): An (n,) array. Observations (Might be different from x_j).
        x_j (np.ndarray): An (m,) array. Observations (Might be different from x_i).
        gamma (float): Gamma parameter for RBF kernel. (Inverse of standard deviation)

    Returns:
        np.ndarray: A (n, m) matrix, where each element is as described above (see equation for K[a, b])

    Note:
        - It is crucial for this function to be vectorized, and not contain for-loops.
            It will be called a lot, and it has to be fast for reasonable run-time.
        - You might find .outer functions useful for this function.
            They apply an operation similar to xx^T (if x is a vector), but not necessarily with multiplication.
            To use it simply append .outer to function. For example: np.add.outer, np.divide.outer
    """
    # raise NotImplementedError("Your Code Goes Here")
    return np.exp(-gamma * np.subtract.outer(x_i, x_j) ** 2)


@problem.tag("hw3-A")
def train(
        x: np.ndarray,
        y: np.ndarray,
        kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
        kernel_param: Union[int, float],
        _lambda: float,
) -> np.ndarray:
    """Trains and returns an alpha vector, that can be used to make predictions.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.

    Returns:
        np.ndarray: Array of shape (n,) containing alpha hat as described in the pdf.
    """
    # raise NotImplementedError("Your Code Goes Here")
    # Solves (K + cI) alpha = y.
    K = kernel_function(x, x, kernel_param)
    return np.linalg.solve(K + _lambda * np.eye(np.min(K.shape)), y)


@problem.tag("hw3-A", start_line=1)
def cross_validation(
        x: np.ndarray,
        y: np.ndarray,
        kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
        kernel_param: Union[int, float],
        _lambda: float,
        num_folds: int,
) -> float:
    """Performs cross validation.

    In a for loop over folds:
        1. Set current fold to be validation, and set all other folds as training set.
        2, Train a function on training set, and then get mean squared error on current fold (validation set).
    Return validation loss averaged over all folds.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        float: Average loss of trained function on validation sets across all folds.
    """
    # Stuff.
    fold_size = len(x) // num_folds

    # raise NotImplementedError("Your Code Goes Here")

    # Split data.
    x_folds = np.array_split(x, num_folds)
    y_folds = np.array_split(y, num_folds)

    # Cross-validation.
    errors = np.zeros(shape=num_folds)
    for i in range(num_folds):
        # Set current fold to be validation set.
        x_valid = x_folds[i]
        y_valid = y_folds[i]

        # Make the other folds the training data.
        x_train = np.concatenate(x_folds[:i] + x_folds[i + 1:])
        y_train = np.concatenate(y_folds[:i] + y_folds[i + 1:])

        # Train the model.
        alpha_hat = train(x=x_train, y=y_train, kernel_function=kernel_function, kernel_param=kernel_param, _lambda=_lambda)

        # Test the model on validation set.
        K = kernel_function(x_train, x_valid, kernel_param)
        y_test = K.T @ alpha_hat

        # Get error.
        errors[i] = np.mean((y_test - y_valid) ** 2)

    # Result.
    res = np.mean(errors)

    return float(res)


@problem.tag("hw3-A")
def rbf_param_search(
        x: np.ndarray, y: np.ndarray, num_folds: int
) -> Tuple[float, float]:
    """
    Parameter search for RBF kernel.

    There are two possible approaches:
        - Grid Search - Fix possible values for lambda, loop over them and record value with the lowest loss.
        - Random Search - Fix number of iterations, during each iteration sample lambda from some distribution and record value with the lowest loss.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        num_folds (int): Number of folds. It should be len(x) for LOO.

    Returns:
        Tuple[float, float]: Tuple containing best performing lambda and gamma pair.

    Note:
        - You do not really need to search over gamma. 1 / (median(dist(x_i, x_j)^2) for all unique pairs x_i, x_j in x
            should be sufficient for this problem. That being said you are more than welcome to do so.
        - If using random search we recommend sampling lambda from distribution 10**i, where i~Unif(-5, -1)
        - If using grid search we recommend choosing possible lambdas to 10**i, where i=linspace(-5, -1)
    """
    # raise NotImplementedError("Your Code Goes Here")

    # Number of values.
    k = 500

    # gamma (feeling lazy)...
    gamma_opt = 1 / (np.median((np.subtract.outer(x, x)) ** 2))

    # Grid search for lambda...
    lam_values = 10 ** np.linspace(-5, -1, k)
    losses = np.zeros(shape=k)
    for i in range(k):
        lam = lam_values[i]
        losses[i] = cross_validation(x=x, y=y, kernel_function=rbf_kernel, kernel_param=gamma_opt, _lambda=lam, num_folds=num_folds)
    lam_opt = lam_values[losses == np.min(losses)][0]

    return lam_opt, gamma_opt


@problem.tag("hw3-A")
def poly_param_search(
        x: np.ndarray, y: np.ndarray, num_folds: int
) -> Tuple[float, int]:
    """
    Parameter search for Poly kernel.

    There are two possible approaches:
        - Grid Search - Fix possible values for lambdas and ds.
            Have nested loop over all possibilities and record value with the lowest loss.
        - Random Search - Fix number of iterations, during each iteration sample lambda, d from some distributions and record value with the lowest loss.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        Tuple[float, int]: Tuple containing best performing lambda and d pair.

    Note:
        - If using random search we recommend sampling lambda from distribution 10**i, where i~Unif(-5, -1)
            and d from distribution [5, 6, ..., 24, 25]
        - If using grid search we recommend choosing possible lambdas to 10**i, where i=linspace(-5, -1)
            and possible d to [5, 6, ..., 24, 25]
    """
    # raise NotImplementedError("Your Code Goes Here")

    # Number of values.
    k_lam = 500
    k_d = 20

    # Grid search for lambda and d.
    lam_values = 10 ** np.linspace(-5, -1, k_lam)
    d_values = np.linspace(5, 25, k_d).astype(int)

    losses = np.zeros(shape=(k_lam, k_d))
    for i, lam in np.ndenumerate(lam_values):
        for j, d in np.ndenumerate(d_values):
            losses[i, j] = cross_validation(x=x, y=y, kernel_function=poly_kernel, kernel_param=d, _lambda=lam, num_folds=num_folds)

    index_opt = np.argwhere(losses == np.min(losses))[0]

    # Get optimal (lambda, d) pair.
    lam_opt = lam_values[index_opt[0]]
    d_opt = d_values[index_opt[1]]

    return lam_opt, d_opt


@problem.tag("hw3-A", start_line=1)
def main():
    """
    Main function of the problem

    It should:
        A. Using x_30, y_30, rbf_param_search and poly_param_search report optimal values for lambda (for rbf), gamma, lambda (for poly) and d.
            Note that x_30, y_30 has been loaded in for you. You do not need to use (x_300, y_300) or (x_1000, y_1000).
        B. For both rbf and poly kernels, train a function using x_30, y_30 and plot predictions on a fine grid

    Note:
        - In part b fine grid can be defined as np.linspace(0, 1, num=100)
        - When plotting you might find that your predictions go into hundreds, causing majority of the plot to look like a flat line.
            To avoid this call plt.ylim(-6, 6).
    """

    # Load data.
    (x_30, y_30), (x_300, y_300), (x_1000, y_1000) = load_dataset("kernel_bootstrap")

    # raise NotImplementedError("Your Code Goes Here")

    # Cross validate to find hyperparameters.
    lam_opt_rbf, gamma_opt_rbf = rbf_param_search(x=x_30, y=y_30, num_folds=len(x_30))
    lam_opt_poly, d_opt_poly = poly_param_search(x=x_30, y=y_30, num_folds=len(x_30))

    print("RBF Values (lambda, gamma):", (lam_opt_rbf, gamma_opt_rbf))
    print("Poly Values (lambda, d):", (lam_opt_poly, d_opt_poly))

    # Train with optimal parameters for RBF.
    alpha_hat_rbf = train(x=x_30, y=y_30, kernel_function=rbf_kernel, kernel_param=gamma_opt_rbf, _lambda=lam_opt_rbf)

    # Train with optimal parameters for Poly.
    alpha_hat_poly = train(x=x_30, y=y_30, kernel_function=poly_kernel, kernel_param=d_opt_poly, _lambda=lam_opt_poly)

    # Plot predictions.
    x_new = np.linspace(0, 1, 1000)
    y_rbf = rbf_kernel(x_30, x_new, gamma_opt_rbf).T @ alpha_hat_rbf
    y_poly = poly_kernel(x_30, x_new, d_opt_poly).T @ alpha_hat_poly

    fig, ax = plt.subplots()
    plt.scatter(x_30, y_30, label="Data")
    plt.plot(x_new, f_true(x=x_new), label="True")
    plt.plot(x_new, y_rbf, label="RBF")
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("A3 (b) : RBF Kernel")
    plt.show()

    fig, ax = plt.subplots()
    plt.scatter(x_30, y_30, label="Data")
    plt.plot(x_new, f_true(x=x_new), label="True")
    plt.plot(x_new, y_poly, label="Poly")
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("A3 (b) : Polynomial Kernel")
    plt.show()


if __name__ == "__main__":
    main()
