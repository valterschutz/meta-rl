"""Utility functions for training and logging."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import wandb


class OneHotLayer(nn.Module):
    """Converts an integer single-element tensor to a one-hot encoded vector."""

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        x_onehot = F.one_hot(x.to(torch.int64), num_classes=self.num_classes).float()
        return x_onehot


def calc_return(t: Tensor, gamma: float, discount_start=0):
    """
    Calculate the return for a single rollout.

    Args:
        t (Tensor): 1-dimensional tensor of rewards.
        gamma (float): Discount factor.
        discount_start (int): Starting exponent for discounting (default: 0).

    Returns:
        float: The total discounted return.
    """

    if t.dim() != 1:
        raise ValueError("Input tensor `t` must be 1-dimensional.")

    # Compute discount factors
    discounts = gamma ** (torch.arange(len(t)) + discount_start)

    # Calculate the discounted return
    return (t * discounts).sum().item()


class DictWrapper:
    """Wraps a normal dict such that its keys can be accessed as attributes."""

    def __init__(self, dict):
        # Set attributes from the dictionary
        for key, value in dict.items():
            setattr(self, key, value)

    def to_dict(self):
        # Optionally, provide a method to convert back to a dictionary
        return self.__dict__


def print_computational_graph(tensor, depth=0):
    """
    Recursively print the computational graph of a tensor.
    """
    if not isinstance(tensor, torch.Tensor):
        # If the object isn't a tensor, just print its type and return.
        print(" " * depth + f"Non-tensor object: {type(tensor)}")
        return

    # Print the current tensor details
    grad_fn = getattr(tensor, "grad_fn", None)
    print(
        " " * depth
        + f"Tensor: {tensor.size()}, requires_grad={tensor.requires_grad}, grad_fn={grad_fn}"
    )

    # If there is no `grad_fn`, this is a leaf tensor (e.g., model parameters or inputs)
    if grad_fn is None:
        return

    # Traverse the inputs to this grad_fn
    try:
        for inp in grad_fn.next_functions:
            if inp[0] is not None:  # inp[0] is the `Function` producing the input
                print_computational_graph(inp[0], depth + 2)
    except AttributeError as e:
        print(" " * depth + f"Error traversing graph: {e}")


class MethodLogger:
    def __init__(self, obj):
        self._wrapped = obj

    def __getattr__(self, name):
        attr = getattr(self._wrapped, name)
        if callable(attr):

            def wrapper(*args, **kwargs):
                # print(f"Method called: {name} | args: {args} | kwargs: {kwargs}")
                print(f"Method {name} called in {self._wrapped.__class__.__name__}")
                return attr(*args, **kwargs)

            return wrapper
        return attr
