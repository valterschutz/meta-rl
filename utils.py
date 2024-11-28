"""Utility functions for training and logging."""

import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb


class OneHotLayer(nn.Module):
    """Converts an integer single-element tensor to a one-hot encoded vector."""

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        x_onehot = F.one_hot(x.to(torch.int64), num_classes=self.num_classes).float()
        return x_onehot


def calc_return(td, gamma):
    """Calculate return for a single rollout"""

    G = 0
    for i in range(len(td)):
        G += td["next", "reward"][i].item() * gamma**i
    return G


class DictWrapper:
    """Wraps a normal dict such that its keys can be accessed as attributes."""

    def __init__(self, dict):
        # Set attributes from the dictionary
        for key, value in dict.items():
            setattr(self, key, value)

    def to_dict(self):
        # Optionally, provide a method to convert back to a dictionary
        return self.__dict__
