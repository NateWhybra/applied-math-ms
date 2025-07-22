if __name__ == "__main__":
    from ISTA import train, get_lambda_range  # type: ignore
else:
    from .ISTA import train

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


@problem.tag("hw2-A", start_line=3)
def main(eta=0.00001, convergence_delta=1e-4, lambda_thresh=0.01):
    # df_train and df_test are pandas dataframes.
    # Make sure you split them into observations and targets.
    df_train, df_test = load_dataset("crime")

    # Get y_train and y_test.
    y_train = df_train.iloc[:, 0].to_numpy()
    y_test = df_test.iloc[:, 0].to_numpy()

    # Get X_train and X_test.
    X_train = df_train.iloc[:, 1:].to_numpy()
    X_test = df_test.iloc[:, 1:].to_numpy()

    # Number of features.
    d = X_train.shape[1]

    # Get lambda values.
    lambda_values, num_lambda = get_lambda_range(X=X_train, y=y_train, c=lambda_thresh)

    # Make storage to save the weights for each lambda.
    all_weights = np.zeros(shape=(d, num_lambda))
    all_bias = np.zeros(shape=num_lambda)
    all_training_error = np.zeros(shape=num_lambda)
    all_test_error = np.zeros(shape=num_lambda)

    # For each value of lambda, train a model.
    model_weights = np.zeros(d)
    model_bias = 0
    for i, _lambda in np.ndenumerate(lambda_values):
        model_weights, model_bias = train(X=X_train, y=y_train, _lambda=_lambda, eta=eta, convergence_delta=convergence_delta, start_weight=model_weights, start_bias=model_bias)
        all_weights[:, i] = model_weights.reshape((d, 1))
        all_bias[i] = model_bias

        # Compute the training squared error.
        all_training_error[i] = np.sum((X_train @ model_weights + model_bias - y_train) ** 2)

        # Compute the test squared error.
        all_test_error[i] = np.sum((X_test @ model_weights + model_bias - y_test) ** 2)

    # For each value of lambda, count how many weights are not zero.
    non_zero_weights = np.sum(all_weights != 0, axis=0)

    # Plot the number of non-zero weights vs lambda (Plot 1).
    fig, ax = plt.subplots()
    ax.scatter(lambda_values, non_zero_weights)
    plt.xscale('log')
    plt.xlabel(r"$\lambda$")
    plt.ylabel("Number of Non-Zero Weights")
    plt.title("Plot 1 (A6)")
    plt.show()

    # Get the indices for agePct12t29, pctWSocSec, pctUrban, agePct65up, and householdsize.
    feature_names = ['agePct12t29', 'pctWSocSec', 'pctUrban', 'agePct65up', 'householdsize']
    # We subtract 1 because our data is missing the first column.
    feature_indices = [df_train.columns.get_loc(name) - 1 for name in feature_names]

    # Get weights for the features.
    # For each feature, there are num_lambda weights. So we will make a list of length 5 containing numpy arrays of length num_lambda.
    feature_weights = []
    num_features = len(feature_names)
    # For each feature.
    for i in range(num_features):
        # For each lambda, grab the weight of feature i.
        current_feature_weights = np.zeros(shape=num_lambda)
        for j, _lambda in np.ndenumerate(lambda_values):
            current_feature_weights[j] = all_weights[feature_indices[i], j]
        feature_weights.append(current_feature_weights)

    # Plot the regularization curves for A6 (d).
    fig, ax = plt.subplots()
    for i in range(num_features):
        ax.plot(lambda_values, feature_weights[i], label=feature_names[i])
    plt.xscale('log')
    plt.xlabel(r'$\lambda$')
    plt.ylabel('Weights')
    plt.title('Plot 2 (A6)')
    plt.legend()
    plt.show()

    # Plot the errors as a function of lambda for A6 (e).
    fig, ax = plt.subplots()
    ax.plot(lambda_values, all_training_error, label='Training Error')
    ax.plot(lambda_values, all_test_error, label='Test Error')
    plt.xscale('log')
    plt.xlabel(r'$\lambda$')
    plt.ylabel('Squared Error')
    plt.title('Plot 3 (A6)')
    plt.legend()
    plt.show()

    # Retrain model with lambda = 30 for A6 (f).
    model_weights, model_bias = train(X=X_train, y=y_train, _lambda=30, eta=eta, convergence_delta=convergence_delta, start_weight=np.zeros(d), start_bias=0)

    # Get the most positive and most negative weight (add 1 because we're going to get the names from the pandas dataframe).
    index_most_positive_weight = np.argmax(model_weights) + 1
    index_most_negative_weight = np.argmin(model_weights) + 1

    # Get most positive/negative features.
    positive_feature = df_train.columns[index_most_positive_weight]
    negative_feature = df_train.columns[index_most_negative_weight]
    print("Feature with most positive weight:", positive_feature)
    print("Feature with most negative weight:", negative_feature)

    # raise NotImplementedError("Your Code Goes Here")




if __name__ == "__main__":
    main()
