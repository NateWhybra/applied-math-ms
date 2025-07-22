if __name__ == "__main__":
    from layers import LinearLayer, ReLULayer, SigmoidLayer
    from losses import MSELossLayer
    from optimizers import SGDOptimizer
    from train import plot_model_guesses, train
else:
    from .layers import LinearLayer, ReLULayer, SigmoidLayer
    from .optimizers import SGDOptimizer
    from .losses import MSELossLayer
    from .train import plot_model_guesses, train

from typing import Any, Dict

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem

RNG = torch.Generator()
RNG.manual_seed(446)


@problem.tag("hw3-A")
def accuracy_score(model: nn.Module, dataloader: DataLoader) -> float:
    """Calculates accuracy of model on dataloader. Returns it as a fraction.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): Dataloader for MSE.
            Each example is a tuple consiting of (observation, target).
            Observation is a 2-d vector of floats.
            Target is also a 2-d vector of floats, but specifically with one being 1.0, while other is 0.0.
            Index of 1.0 in target corresponds to the true class.

    Returns:
        float: Vanilla python float resprenting accuracy of the model on given dataset/dataloader.
            In range [0, 1].

    Note:
        - For a single-element tensor you can use .item() to cast it to a float.
        - This is similar to CrossEntropy accuracy_score function,
            but there will be differences due to slightly different targets in dataloaders.
    """
    # raise NotImplementedError("Your Code Goes Here")

    # Turn on evaluation mode.
    model.eval()

    # Initialize counts.
    correct = 0
    total = 0

    # Don't need to use gradients.
    with torch.no_grad():
        for observation, target in dataloader:
            # Forward pass.
            output = model(observation)

            # Check this later.
            predicted = torch.argmax(output, dim=1)
            true_labels = torch.argmax(target, dim=1)

            # Count how many are correct and cast as float.
            correct += (predicted == true_labels).sum().item()

            # Count how many data points we have.
            total += target.size(0)

    # Compute accuracy.
    accuracy = correct / total

    return accuracy


@problem.tag("hw3-A")
def mse_parameter_search(
        dataset_train: TensorDataset, dataset_val: TensorDataset,
        batch_size=100, epochs=1) -> Dict[str, Any]:
    """
    Main subroutine of the MSE problem.
    It's goal is to perform a search over hyperparameters, and return a dictionary containing training history of models, as well as models themselves.

    Models to check (please try them in this order):
        - Linear Regression Model (1)
        - Network with one hidden layer of size 2 and sigmoid activation function after the hidden layer (2)
        - Network with one hidden layer of size 2 and ReLU activation function after the hidden layer (3)
        - Network with two hidden layers (each with size 2) (4)
            and Sigmoid, ReLU activation function after corresponding hidden layers
        - Network with two hidden layers (each with size 2) (5)
            and ReLU, Sigmoid activation function after corresponding hidden layers

    Notes:
        - When choosing the number of epochs, consider effect of other hyperparameters on it.
            For example as learning rate gets smaller you will need more epochs to converge.

    Args:
        dataset_train (TensorDataset): Training dataset.
        dataset_val (TensorDataset): Validation dataset.

    Returns:
        Dict[str, Any]: Dictionary/Map containing history of training of all models.
            You are free to employ any structure of this dictionary, but we suggest the following:
            {
                name_of_model: {
                    "train": Per epoch losses of model on train set,
                    "val": Per epoch losses of model on validation set,
                    "model": Actual PyTorch model (type: nn.Module),
                }
            }
    """
    # raise NotImplementedError("Your Code Goes Here")

    # Make space for output (model histories).
    model_hists = {"Model 1": {"train": None, "val": None, "model": None},
                   "Model 2": {"train": None, "val": None, "model": None},
                   "Model 3": {"train": None, "val": None, "model": None},
                   "Model 4": {"train": None, "val": None, "model": None},
                   "Model 5": {"train": None, "val": None, "model": None}}

    # Get data loader for training data / validation data.
    train_loader = DataLoader(dataset_train, batch_size=batch_size)
    val_loader = DataLoader(dataset_val, batch_size=batch_size)

    # Make Linear Model (1).
    model_1 = LinearLayer(dim_in=2, dim_out=1)
    optimizer_1 = SGDOptimizer(model_1.parameters(), lr=1e-3)
    history_1 = train(train_loader=train_loader, model=model_1, criterion=MSELossLayer(), optimizer=optimizer_1, val_loader=val_loader, epochs=epochs, batch_size=batch_size)
    history_1["model"] = model_1
    model_hists["Model 1"] = history_1

    # Network with one hidden layer of size 2 and sigmoid activation function after the hidden layer (2).
    hidden_2 = LinearLayer(dim_in=2, dim_out=1)
    output_2 = SigmoidLayer()
    model_2 = torch.nn.Sequential(hidden_2, output_2)
    optimizer_2 = SGDOptimizer(model_2.parameters(), lr=1e-3)
    history_2 = train(train_loader=train_loader, model=model_2, criterion=MSELossLayer(), optimizer=optimizer_2, val_loader=val_loader, epochs=epochs, batch_size=batch_size)
    history_2["model"] = model_2
    model_hists["Model 2"] = history_2

    # Network with one hidden layer of size 2 and ReLU activation function after the hidden layer (3).
    hidden_3 = LinearLayer(dim_in=2, dim_out=1)
    output_3 = ReLULayer()
    model_3 = torch.nn.Sequential(hidden_3, output_3)
    optimizer_3 = SGDOptimizer(model_3.parameters(), lr=1e-3)
    history_3 = train(train_loader=train_loader, model=model_3, criterion=MSELossLayer(), optimizer=optimizer_3, val_loader=val_loader, epochs=epochs, batch_size=batch_size)
    history_3["model"] = model_3
    model_hists["Model 3"] = history_3

    # Network with two hidden layers (each with size 2) and Sigmoid, ReLU activation function after corresponding hidden layers (4).
    hidden_41 = LinearLayer(dim_in=2, dim_out=2)
    output_41 = SigmoidLayer()
    hidden_42 = LinearLayer(dim_in=2, dim_out=1)
    output_42 = ReLULayer()
    model_4 = torch.nn.Sequential(hidden_41, output_41, hidden_42, output_42)
    optimizer_4 = SGDOptimizer(model_4.parameters(), lr=1e-3)
    history_4 = train(train_loader=train_loader, model=model_4, criterion=MSELossLayer(), optimizer=optimizer_4, val_loader=val_loader, epochs=epochs, batch_size=batch_size)
    history_4["model"] = model_4
    model_hists["Model 4"] = history_4

    # Network with two hidden layers (each with size 2) and ReLU, Sigmoid activation function after corresponding hidden layers (5).
    hidden_51 = LinearLayer(dim_in=2, dim_out=2)
    output_51 = ReLULayer()
    hidden_52 = LinearLayer(dim_in=2, dim_out=1)
    output_52 = SigmoidLayer()
    model_5 = torch.nn.Sequential(hidden_51, output_51, hidden_52, output_52)
    optimizer_5 = SGDOptimizer(model_5.parameters(), lr=1e-3)
    history_5 = train(train_loader=train_loader, model=model_5, criterion=MSELossLayer(), optimizer=optimizer_5, val_loader=val_loader, epochs=epochs, batch_size=batch_size)
    history_5["model"] = model_5
    model_hists["Model 5"] = history_5

    return model_hists


@problem.tag("hw3-A", start_line=11)
def main(epochs=1000, batch_size=100):
    """
    Main function of the MSE problem.
    It should:
        1. Call mse_parameter_search routine and get dictionary for each model architecture/configuration.
        2. Plot Train and Validation losses for each model all on single plot (it should be 10 lines total).
            x-axis should be epochs, y-axis should me MSE loss, REMEMBER to add legend
        3. Choose and report the best model configuration based on validation losses.
            In particular you should choose a model that achieved the lowest validation loss at ANY point during the training.
        4. Plot best model guesses on test set (using plot_model_guesses function from train file)
        5. Report accuracy of the model on test set.

    Starter code loads dataset, converts it into PyTorch Datasets, and those into DataLoaders.
    You should use these dataloaders, for the best experience with PyTorch.
    """
    (x, y), (x_val, y_val), (x_test, y_test) = load_dataset("xor")

    dataset_train = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(to_one_hot(y)))
    dataset_val = TensorDataset(
        torch.from_numpy(x_val).float(), torch.from_numpy(to_one_hot(y_val))
    )
    dataset_test = TensorDataset(
        torch.from_numpy(x_test).float(), torch.from_numpy(to_one_hot(y_test))
    )
    mse_configs = mse_parameter_search(dataset_train, dataset_val, epochs=epochs, batch_size=batch_size)

    # raise NotImplementedError("Your Code Goes Here")

    # Names of models.
    names = mse_configs.keys()

    # Plot everything.
    fig, ax = plt.subplots()
    for model_name in names:
        loss_train = mse_configs[model_name]["train"]
        loss_val = mse_configs[model_name]['val']

        if loss_train:
            plt.plot(np.arange(epochs) + 1, loss_train, label=model_name + " - Train")
        if loss_val:
            plt.plot(np.arange(epochs) + 1, loss_val, label=model_name + " - Validation")

        loss_train = None
        loss_val = None

    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()


def to_one_hot(a: np.ndarray) -> np.ndarray:
    """Helper function. Converts data from categorical to one-hot encoded.

    Args:
        a (np.ndarray): Input array of integers with shape (n,).

    Returns:
        np.ndarray: Array with shape (n, c), where c is maximal element of a.
            Each element of a, has a corresponding one-hot encoded vector of length c.
    """
    r = np.zeros((len(a), 2))
    r[np.arange(len(a)), a] = 1
    return r


if __name__ == "__main__":
    main(epochs=50, batch_size=25)
