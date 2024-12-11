import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict.nn.distributions import NormalParamExtractor


class MetaPolicyNet(nn.Module):
    def __init__(self, n_states, hidden_units, device):
        super().__init__()
        self.net = nn.Sequential(
            # nn.Linear(n_states, hidden_units),
            # nn.Tanh(),
            # nn.Linear(hidden_units, hidden_units),
            # nn.Tanh(),
            # nn.Linear(hidden_units, 2),
            nn.Linear(n_states, 2),
        ).to(device)
        self.normal_params = (
            NormalParamExtractor()
        )  # TODO: This seems to add constants to the variance...

    def forward(self, *args):
        x = torch.cat(args, dim=-1)
        x = self.net(x)  # (batch, 2 * n_actions)
        loc, scale = self.normal_params(x)
        loc = F.sigmoid(loc)
        return loc, scale


class MetaQValueNet(nn.Module):
    def __init__(self, n_states, hidden_units, device):
        super().__init__()
        self.net = nn.Sequential(
            # nn.Linear(n_states + 1, hidden_units),
            # nn.Tanh(),
            # nn.Linear(hidden_units, hidden_units),
            # nn.Tanh(),
            # nn.Linear(hidden_units, 1),
            nn.Linear(n_states + 1, 1)
        ).to(device)

    def forward(self, *args):
        x = torch.cat(args, dim=-1)
        return self.net(x)


class MetaValueNet(nn.Module):
    def __init__(self, n_states, hidden_units, device):
        super().__init__()
        self.net = nn.Sequential(
            # nn.Linear(n_states + 1, hidden_units),
            # nn.Tanh(),
            # nn.Linear(hidden_units, hidden_units),
            # nn.Tanh(),
            # nn.Linear(hidden_units, 1),
            nn.Linear(n_states, 1)
        ).to(device)

    def forward(self, *args):
        x = torch.cat(args, dim=-1)
        return self.net(x)
