import torch
import torch.nn as nn


class MetaPolicyNet(nn.Module):
    def __init__(self, hidden_units, n_states, n_outputs, device):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden_units),
            # nn.Tanh(),
            nn.ReLU(),
            nn.Linear(hidden_units, n_outputs),
            nn.Sigmoid(),
            # nn.ReLU(),  # If using Beta distribution
            # NormalParamExtractor(),
        ).to(device)

    def forward(self, *args):
        x = torch.cat(args, dim=-1)
        return self.net(x)


class MetaValueNet(nn.Module):
    def __init__(self, hidden_units, n_states, device):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, 1),
        ).to(device)

    def forward(self, *args):
        x = torch.cat(args, dim=-1)
        return self.net(x)


class MetaQValueNet(nn.Module):
    def __init__(self, hidden_units, n_states, n_actions, device):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, n_actions),
        ).to(device)

    def forward(self, *args):
        x = torch.cat(args, dim=-1)
        return self.net(x)
