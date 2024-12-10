import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict.nn.distributions import NormalParamExtractor


class MetaPolicyNet(nn.Module):
    def __init__(self, hidden_net, hidden_units, n_actions, device):
        super().__init__()
        self.net = nn.Sequential(
            hidden_net, nn.Tanh(), nn.Linear(hidden_units, 2 * n_actions)
        ).to(device)
        self.normal_params = NormalParamExtractor()

    def forward(self, *args):
        x = torch.cat(args, dim=-1)
        x = self.net(x)  # (batch, 2 * n_actions)
        loc, scale = self.normal_params(x)
        loc = F.sigmoid(loc)
        return loc, scale


class MetaQValueNet(nn.Module):
    def __init__(self, hidden_net, hidden_units, n_actions, device):
        super().__init__()
        self.net = nn.Sequential(
            hidden_net,
            nn.Tanh(),
            nn.Linear(hidden_units, n_actions),
        ).to(device)

    def forward(self, *args):
        x = torch.cat(args, dim=-1)
        return self.net(x)
