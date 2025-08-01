from typing import Optional

import torch
from torch import nn

from utils import problem


class LinearLayer(nn.Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(
            self, dim_in: int, dim_out: int, generator: Optional[torch.Generator] = None
    ):
        """Linear Layer, which performs calculation of: x @ weight + bias

        In constructor you should initialize weight and bias according to dimensions provided.
        You should use torch.randn function to initialize them by normal distribution, and provide the generator if it's defined.

        Both weight and bias should be of torch's type float.
        Additionally, for optimizer to work properly you will want to wrap both weight and bias in nn.Parameter.

        Args:
            dim_in (int): Number of features in data input.
            dim_out (int): Number of features output data should have.
            generator (Optional[torch.Generator], optional): Generator to use when creating weight and bias.
                If defined it should be passed into torch.randn function.
                Defaults to None.

        Note:
            - YOU ARE NOT ALLOWED to use torch.nn.Linear (or it's functional counterparts) in this class
            - Make use of pytorch documentation: https://pytorch.org/docs/stable/index.html
        """
        super().__init__()
        # raise NotImplementedError("Your Code Goes Here")

        # Initialize weights and biases.
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.weight = torch.nn.parameter.Parameter(torch.randn(size=(dim_in, dim_out)).float())
        self.bias = torch.nn.parameter.Parameter(torch.randn(size=(1, dim_out)).float())

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Actually perform multiplication x @ weight + bias

        Args:
            x (torch.Tensor): More specifically a torch.FloatTensor, with shape of (n, dim_in).
                Input data.

        Returns:
            torch.Tensor: More specifically a torch.FloatTensor, with shape of (n, dim_out).
                Output data.
        """
        # raise NotImplementedError("Your Code Goes Here")
        return x @ self.weight + self.bias
