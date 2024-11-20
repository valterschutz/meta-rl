from collections import defaultdict
from tqdm import tqdm
from datetime import datetime

from torchrl.envs.utils import (
    check_env_specs,
    step_mdp,
    set_exploration_type,
    ExplorationType,
)
from torchrl.envs.transforms import TransformedEnv, Compose, StepCounter
from torchrl.modules.tensordict_module.actors import QValueActor
from torchrl.modules.tensordict_module import ProbabilisticActor, ValueOperator
from torchrl.modules import EGreedyModule
from torchrl.data import OneHot, ReplayBuffer, PrioritizedReplayBuffer
from torchrl.data.replay_buffers import LazyTensorStorage, SamplerWithoutReplacement
from torchrl.objectives import (
    DQNLoss,
    SoftUpdate,
    ClipPPOLoss,
    ReinforceLoss,
    DiscreteSACLoss,
)
from torchrl.objectives.value import GAE
from torchrl.collectors import SyncDataCollector
from torchrl.trainers import Trainer
from torchrl.record.loggers.csv import CSVLogger
from torchrl.record.loggers.wandb import WandbLogger


from tensordict import TensorDict
from tensordict.nn import TensorDictSequential, TensorDictModule
from tensordict.nn.distributions import (
    NormalParamExtractor,
    OneHotCategorical,
    # Categorical,
)
from torch.distributions import Categorical
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import numpy as np

import wandb

from env import ToyEnv
from value_iteration import value_iteration


class OneHotLayer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        # Convert the integer state to one-hot encoded vector
        x_onehot = F.one_hot(x.to(torch.int64), num_classes=self.num_classes).float()
        return x_onehot


device = torch.device("cpu")
total_frames = 1_000
batch_size = 100
buffer_size = 1000
sub_batch_size = 10
num_optim_epochs = 10
gamma = 1
n_actions = 4
# max_grad_norm = 1
lr = 1e-2
times_to_eval = 10
eval_every_n_epoch = (total_frames // batch_size) // times_to_eval
max_rollout_steps = 100
alpha_init = 1
tau = 0.005  # For updating target networks
# lmbda = 0.9

left_reward = -2.0
right_reward = 1.0
down_reward = -3.0
up_reward = 3.0
punishment = 0.1
n_pos = 8
big_reward = 10
env = ToyEnv(
    left_reward=left_reward,
    right_reward=right_reward,
    down_reward=down_reward,
    up_reward=up_reward,
    n_pos=n_pos,
    big_reward=big_reward,
    punishment=punishment,
    random_start=False,
).to(device)
# Optimal return is 4, if moving right from starting state
# add stepcount transform
env = TransformedEnv(env, Compose(StepCounter()))
dummy_td = env.rollout(3)
next_dummy_td = env.step(dummy_td)

# do a rollout and see how long it is
td = env.rollout(1000)
# print(f"{td=}")
td_step = env.rand_step(td)
# print(f"{td_step=}")

check_env_specs(env)

hidden_units = 32

actor_net = nn.Sequential(
    OneHotLayer(num_classes=n_pos),
    nn.Linear(n_pos, hidden_units),
    nn.Tanh(),
    nn.Linear(hidden_units, n_actions),
).to(device)
policy_module = TensorDictModule(actor_net, in_keys=["pos"], out_keys=["logits"])
policy_module = ProbabilisticActor(
    module=policy_module,
    spec=env.action_spec,
    in_keys=["logits"],
    distribution_class=Categorical,
    return_log_prob=True,
)

qvalue_net = nn.Sequential(
    OneHotLayer(num_classes=n_pos),
    nn.Linear(n_pos, hidden_units),
    nn.Tanh(),
    nn.Linear(hidden_units, n_actions),
).to(device)
qvalue_module = ValueOperator(qvalue_net, in_keys=["pos"], out_keys=["action_value"])

# advantage_module = GAE(
#     gamma=gamma, lmbda=lmbda, value_network=qvalue_module, average_gae=True
# )


# loss_module = ReinforceLoss(policy_module, value_module)
loss_module = DiscreteSACLoss(
    policy_module,
    qvalue_module,
    action_space=env.action_spec,
    num_actions=n_actions,
    loss_function="l2",
    alpha_init=alpha_init,
)
target_updater = SoftUpdate(loss_module, tau=tau)

optim = torch.optim.Adam(loss_module.parameters(), lr=lr)

replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(max_size=buffer_size, device=device),
    sampler=SamplerWithoutReplacement(),
)

collector = SyncDataCollector(
    env,
    policy_module,
    frames_per_batch=batch_size,
    total_frames=total_frames,
    split_trajs=False,
    device=device,
)

all_pos = torch.arange(n_pos)
td_all_pos = TensorDict(
    {"pos": all_pos, "step_count": torch.zeros_like(all_pos)},
    batch_size=[n_pos],
).to(device)

wandb.login()
wandb.init(
    project="toy_sac",
    name=f"toy_sac|{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    config={
        "buffer_size": buffer_size,
        "frames_per_batch": batch_size,
        "sub_batch_size": sub_batch_size,
        "total_frames": total_frames,
        "num_optim_epochs": num_optim_epochs,
        "gamma": gamma,
        "n_actions": n_actions,
        # "max_grad_norm": max_grad_norm,
        "lr": lr,
        "left_reward": left_reward,
        "right_reward": right_reward,
        "down_reward": down_reward,
        "up_reward": up_reward,
        # "x": x,
        # "y": y,
        "n_pos": n_pos,
        "big_reward": big_reward,
        "punishment": punishment,
    },
)

pbar = tqdm(total=total_frames)

for i, td in enumerate(collector):
    replay_buffer.extend(td)
    for _ in range(num_optim_epochs):
        # advantage_module(td)
        # for _ in range(batch_size // sub_batch_size):
        subdata = replay_buffer.sample(sub_batch_size)
        loss_vals = loss_module(subdata)
        # print(f"{loss_vals=}")
        # fail
        loss = (
            loss_vals["loss_actor"] + loss_vals["loss_qvalue"] + loss_vals["loss_alpha"]
        )
        loss.backward()
        # nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
        optim.step()
        optim.zero_grad()
        target_updater.step()
        # wandb.log({"loss": loss_value.item(), "grad_norm": grad_norm.max().item()})
    pbar.update(td.numel())
    wandb.log(
        {
            "reward": td["next", "reward"].float().mean().item(),
            "loss_actor": loss_vals["loss_actor"].item(),
            "loss_qvalue": loss_vals["loss_qvalue"].item(),
            "loss_alpha": loss_vals["loss_alpha"].item(),
            "loss": loss.item(),
            "state distribution": wandb.Histogram(td["pos"].cpu().numpy()),
            # "state distribution": wandb.Histogram(np.random.randn(10, 10)),
            "alpha": loss_vals["alpha"].item(),
            "entropy": loss_vals["entropy"].item(),
            # "grad_norm": sum(p**2 for p in loss_module.parameters()).sqrt().item(),
        }
    )
    # wandb.Histogram(pos=td["pos"].cpu())
    if i % eval_every_n_epoch == 0:
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            # execute a rollout with the (greedy) policy and calculate return
            eval_rollout = env.rollout(max_rollout_steps, policy_module)
            wandb.log(
                {
                    "eval return": eval_rollout["next", "reward"].sum().item(),
                    "eval length": eval_rollout["step_count"].max().item(),
                }
            )

# After training, do a single rollout and print all states and actions
with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
    td = env.rollout(max_rollout_steps, policy_module)
    for i in range(len(td)):
        print(
            f"Step {i}: pos={td['pos'][i].item()}, action={td['action'][i].item()}, reward={td['next','reward'][i].item()} done={td['next','done'][i].item()}"
        )
