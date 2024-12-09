import torch
import torch.nn as nn


class MetaPolicyNet(nn.Module):
    def __init__(self, hidden_net, hidden_units, n_actions, device):
        super().__init__()
        self.net = nn.Sequential(
            hidden_net, nn.Tanh(), nn.Linear(hidden_units, n_actions), nn.Sigmoid()
        ).to(device)

    def forward(self, *args):
        x = torch.cat(args, dim=-1)
        return self.net(x)


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
