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
from torchrl.objectives import DQNLoss, SoftUpdate, ClipPPOLoss, ReinforceLoss
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
total_frames = 100_000
batch_size = 1000
buffer_size = batch_size
sub_batch_size = 64
num_optim_epochs = 10
gamma = 1
n_actions = 4
max_grad_norm = 1
lr = 1e-4
eval_every_n_epoch = 1
max_rollout_steps = 1000
lmbda = 0.9

# x = 1
# y = 3
x = 0
y = 0
punishment = False
n_pos = 8
big_reward = 10
env = ToyEnv(x, y, n_pos, big_reward, punishment=punishment, random_start=False)
# Optimal return is 4, if moving right from starting state
# add stepcount transform
env = TransformedEnv(env, Compose(StepCounter()))
# dummy_td = env.reset()
dummy_td = env.rollout(3)
next_dummy_td = env.step(dummy_td)
# print(f"{next_dummy_td=}")
# fail

# do a rollout and see how long it is
td = env.rollout(1000)
# print(f"{td=}")
td_step = env.rand_step(td)
# print(f"{td_step=}")

# fail

check_env_specs(env)

# fail


actor_net = nn.Sequential(
    OneHotLayer(num_classes=n_pos),
    nn.Linear(n_pos, n_actions),
).to(device)
policy_module = TensorDictModule(actor_net, in_keys=["pos"], out_keys=["logits"])
policy_module = ProbabilisticActor(
    module=policy_module,
    spec=env.action_spec,
    # in_keys=["loc", "scale"],
    in_keys=["logits"],
    distribution_class=Categorical,
    return_log_prob=True,
)

# td = env.reset()
# print(f"Before policy: {td=}")
# td = policy_module(td)
# print(f"After policy: {td=}")

# fail

hidden_units = 8
value_net = nn.Sequential(
    OneHotLayer(num_classes=n_pos),
    nn.Linear(n_pos, hidden_units),
    nn.Tanh(),
    nn.Linear(hidden_units, 1),
)
value_module = ValueOperator(value_net, in_keys=["pos"], out_keys=["state_value"])
advantage_module = GAE(
    gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True
)

# td = env.reset()
# print(f"Before value: {td=}")
# td = value_module(td)
# print(f"After value: {td=}")
# td = env.rand_step(td)
# print(f"After step: {td=}")
# fail

# print("Running policy:", policy_module(env.reset()))
# print("Running value:", value_module(env.reset()))
# print(f"before action: {dummy_td=}")
# dummy_td = policy_module(dummy_td)
# print(f"after action: {dummy_td=}")
# print(f"{dummy_td['action']=}")
# td = env.rollout(10, policy_module)
# print(f"{td=}")
# fail

# loss_module = ClipPPOLoss(
#     actor_network=policy_module,
#     critic_network=value_module,
#     clip_epsilon=clip_epsilon,
#     entropy_bonus=bool(entropy_eps),
#     entropy_coef=entropy_eps,
#     # these keys match by default but we set this for completeness
#     critic_coef=1.0,
#     loss_critic_type="smooth_l1",
# )
loss_module = ReinforceLoss(policy_module, value_module)
# dummy_td = env.rollout(10, policy_module)
# print(f"{dummy_td=}")
# print(f"{loss_module(dummy_td)=}")
# fail

optim = torch.optim.Adam(loss_module.parameters(), lr=lr)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     optim, total_frames // sub_batch_size, 0.0
# )

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
    {"pos": all_pos},
    batch_size=[n_pos],
).to(device)

wandb.login()
wandb.init(
    project="toy_reinforce",
    name=f"toy_reinforce|{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    config={
        "buffer_size": buffer_size,
        "frames_per_batch": batch_size,
        "sub_batch_size": sub_batch_size,
        "total_frames": total_frames,
        "num_optim_epochs": num_optim_epochs,
        "gamma": gamma,
        "n_actions": n_actions,
        "max_grad_norm": max_grad_norm,
        "lr": lr,
        "x": x,
        "y": y,
        "n_pos": n_pos,
        "big_reward": big_reward,
        "punishment": punishment,
    },
)

pbar = tqdm(total=total_frames)

for i, td in enumerate(collector):
    for _ in range(num_optim_epochs):
        advantage_module(td)
        replay_buffer.extend(td)
        for _ in range(batch_size // sub_batch_size):
            subdata = replay_buffer.sample(sub_batch_size)
            loss_vals = loss_module(subdata)
            loss_value = loss_vals["loss_actor"] + loss_vals["loss_value"]
            loss_value.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                loss_module.parameters(), max_grad_norm
            )
            optim.step()
            optim.zero_grad()
            # wandb.log({"loss": loss_value.item(), "grad_norm": grad_norm.max().item()})
    pbar.update(td.numel())
    wandb.log(
        {
            "reward": td["next", "reward"].float().mean().item(),
            "loss": loss_value.item(),
            "grad_norm": grad_norm.max().item(),
        }
    )
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
            f"Step {i}: pos={td['pos'][i].item()}, action={td['action'][i].item()}, reward={td['next','reward'][i].item()} done={td['done'][i].item()}"
        )
