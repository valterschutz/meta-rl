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
    out_keys=["action"],
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

td = TensorDict(
    state=torch.tensor([[0.1]]),
    action=torch.tensor([[0.3]]),
    next=TensorDict(
        state=torch.tensor([[0.2]]),
        reward=torch.tensor([[0.5]]),
        done=torch.tensor([[False]]),
    ),
)

loss_td = loss_module(td)
loss = loss_td["loss_actor"] + loss_td["loss_qvalue"] + loss_td["loss_alpha"]

# Print policy gradients before backprop
print(f"Policy gradients before backprop:")
for param in policy_module.parameters():
    print(param.grad)
# Print qvalue gradients before backprop
print(f"Q-value gradients before backprop:")
for param in qvalue_module.parameters():
    print(param.grad)

# Backprop
loss.backward()

# Print policy gradients after backprop
print(f"Policy gradients after backprop:")
for param in policy_module.parameters():
    print(param.grad)
# Print qvalue gradients after backprop
print(f"Q-value gradients after backprop:")
for param in qvalue_module.parameters():
    print(param.grad)
