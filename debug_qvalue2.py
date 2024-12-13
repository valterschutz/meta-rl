import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import InteractionType, TensorDictModule
from torchrl.modules import TanhNormal
from torchrl.modules.distributions import NormalParamExtractor
from torchrl.modules.tensordict_module import ProbabilisticActor, ValueOperator
from torchrl.objectives import SACLoss

# Policy
actor_net = nn.Sequential(
    nn.Linear(1, 2),
    NormalParamExtractor(),
)

policy_module = TensorDictModule(
    actor_net, in_keys=["state"], out_keys=["loc", "scale"]
)
policy_module = ProbabilisticActor(
    module=policy_module,
    in_keys=["loc", "scale"],
    # out_keys=["action"],
    distribution_class=TanhNormal,
    default_interaction_type=InteractionType.RANDOM,
    return_log_prob=True,
)


# QValue
class QValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 1))

    def forward(self, state, action):
        x = torch.cat((state, action), dim=-1)
        return self.net(x)


qvalue_net = QValueNet()
qvalue_module = ValueOperator(
    qvalue_net,
    in_keys=["state", "action"],
    out_keys=["state_action_value"],
)

loss_module = SACLoss(
    actor_network=policy_module,
    qvalue_network=qvalue_module,
)
loss_module.make_value_estimator(gamma=0.9)

optim = torch.optim.Adam(loss_module.parameters(), lr=1e-1)
# NOTE guess you didn't care about target update right?

td = TensorDict(
    {
        "state": torch.tensor([[0.1]]),
        "action": torch.tensor([[0.3]]),
        ("next", "state"): torch.tensor([[0.2]]),
        ("next", "reward"): torch.tensor([[0.5]]),
        ("next", "done"): torch.tensor([[False]]),
    }
)

print(f"Q-value parameters before backprop:")
for param in loss_module.qvalue_network_params.parameters():
    print(param)

# Backprop
# for _ in range(10):
loss_td = loss_module(td)
loss = loss_td["loss_actor"] + loss_td["loss_qvalue"] + loss_td["loss_alpha"]
loss.backward()
optim.step()
optim.zero_grad()

print(f"Q-value parameters after backprop:")
for param in loss_module.qvalue_network_params.parameters():
    print(param)
