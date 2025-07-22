"""
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
"""

from typing import Tuple

import numpy as np

from utils import problem


class PolynomialRegression:
    @problem.tag("hw1-A", start_line=5)
    def __init__(self, degree: int = 1, reg_lambda: float = 1e-8):
        """Constructor
        """
        self.degree: int = degree
        self.reg_lambda: float = reg_lambda
        # Fill in with matrix with the correct shape
        self.weight: np.ndarray = None  # type: ignore
        # You can add additional fields.
        self.mean = 0
        self.std = 0

        # raise NotImplementedError("Your Code Goes Here")

    @staticmethod
    @problem.tag("hw1-A")
    def polyfeatures(X: np.ndarray, degree: int) -> np.ndarray:
        """
        Expands the given X into an (n, degree) array of polynomial features of degree = degree.

        Args:
            X (np.ndarray): Array of shape (n, 1).
            degree (int): Positive integer defining maximum power to include.

        Returns:
            np.ndarray: A (n, degree) numpy array, with each row comprising of
                X, X * X, X ** 3, ... up to the degree^th power of X.
                Note that the returned matrix will not include the zero-th power.

        """
        # raise NotImplementedError("Your Code Goes Here")
        powers = np.arange(degree)
        n = X.shape[0]
        A = np.zeros(shape=(n, degree))
        A[:, powers] = X ** (powers + 1)

        return A

    @problem.tag("hw1-A")
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Trains the model, and saves learned weight in self.weight

        Args:
            X (np.ndarray): Array of shape (n, 1) with observations.
            y (np.ndarray): Array of shape (n, 1) with targets.

        Note:
            You will need to apply polynomial expansion and data standardization first.
        """
        # raise NotImplementedError("Your Code Goes Here")

        # Grab feature matrix without bias column.
        F = self.polyfeatures(X=X, degree=self.degree)
        self.mean = F.mean(axis=0)
        self.std = F.std(axis=0)
        F = (F - self.mean) / self.std

        # Insert bias column.
        b = np.ones_like(y)
        F = np.column_stack([b, F])
        d = self.degree + 1

        # Solve for weights.
        A = (F.T @ F + self.reg_lambda * np.eye(d))
        self.weight = np.linalg.solve(A, F.T @ y)

    @problem.tag("hw1-A")
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Use the trained model to predict values for each instance in X.

        Args:
            X (np.ndarray): Array of shape (n, 1) with observations.

        Returns:
            np.ndarray: Array of shape (n, 1) with predictions.
        """
        # raise NotImplementedError("Your Code Goes Here")

        # Grab feature matrix without bias column.
        F = self.polyfeatures(X=X, degree=self.degree)
        F = (F - self.mean) / self.std

        # Insert bias column.
        b = np.ones(shape=X.shape[0])
        F = np.column_stack([b, F])

        # Each value in y_pred is a dot product.
        y_pred = np.dot(F, self.weight)

        return y_pred


@problem.tag("hw1-A")
def mean_squared_error(a: np.ndarray, b: np.ndarray) -> float:
    """Given two arrays: a and b, both of shape (n, 1) calculate a mean squared error.

    Args:
        a (np.ndarray): Array of shape (n, 1)
        b (np.ndarray): Array of shape (n, 1)

    Returns:
        float: mean squared error between a and b.
    """
    # raise NotImplementedError("Your Code Goes Here")
    return ((a - b) ** 2).sum() / (a.shape[0])


@problem.tag("hw1-A", start_line=5)
def learningCurve(
        Xtrain: np.ndarray,
        Ytrain: np.ndarray,
        Xtest: np.ndarray,
        Ytest: np.ndarray,
        reg_lambda: float,
        degree: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute learning curves.

    Args:
        Xtrain (np.ndarray): Training observations, shape: (n, 1)
        Ytrain (np.ndarray): Training targets, shape: (n, 1)
        Xtest (np.ndarray): Testing observations, shape: (n, 1)
        Ytest (np.ndarray): Testing targets, shape: (n, 1)
        reg_lambda (float): Regularization factor
        degree (int): Polynomial degree

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing:
            1. errorTrain -- errorTrain[i] is the training mean squared error using model trained by Xtrain[0:(i+1)]
            2. errorTest -- errorTest[i] is the testing mean squared error using model trained by Xtrain[0:(i+1)]

    Note:
        - For errorTrain[i] only calculate error on Xtrain[0:(i+1)], since this is the data used for training.
            THIS DOES NOT APPLY TO errorTest.
        - errorTrain[0:1] and errorTest[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    """

    # Make storage.
    n = len(Xtrain)
    errorTrain = np.zeros(n)
    errorTest = np.zeros(n)

    # Fill in errorTrain and errorTest arrays
    # raise NotImplementedError("Your Code Goes Here")
    # Compute training error.
    for i in range(n):
        # Partition data.
        currXTrain = Xtrain[0:i + 1]
        currYTrain = Ytrain[0:i + 1]

        # Train model.
        model = PolynomialRegression(degree=degree, reg_lambda=reg_lambda)
        model.fit(X=currXTrain, y=currYTrain)

        # Get training and test predictions.
        pred_train = model.predict(currXTrain)

        pred_test = model.predict(Xtest)

        # Compute errors.
        errorTrain[i] = mean_squared_error(a=currYTrain, b=pred_train)
        errorTest[i] = mean_squared_error(a=Ytest, b=pred_test)

    return errorTrain, errorTest
