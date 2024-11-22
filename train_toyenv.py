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
from torch.distributions import Categorical, Bernoulli
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import numpy as np

import wandb

from env import ToyEnv
from value_iteration import value_iteration


n_meta_episodes = 10
device = torch.device("cpu")
total_frames = 5_000
batch_size = 100
buffer_size = batch_size  # since on-policy
sub_batch_size = 20
num_optim_epochs = 10
gamma = 0.98
lmbda = 0.96
n_actions = 4
max_grad_norm = 1
lr = 1e-1
times_to_eval = 10
eval_every_n_epoch = (total_frames // batch_size) // times_to_eval
clip_epsilon = 0.2
n_pos = 20
max_rollout_steps = n_pos * 3
optimal_return = 0.1
gap = 0.1
big_reward = 10.0

# (n_pos-1)*x + big_reward = optimal_return
x = (optimal_return - big_reward) / (n_pos - 1)
# (n_pos-1)/2*y + big_reward = optimal_return - gap
y = (optimal_return - gap - big_reward) / ((n_pos - 1) / 2)

# Base env
base_env = get_base_env(
    left_reward=x,
    right_reward=x,
    down_reward=y,
    up_reward=y,
    n_pos=n_pos,
    big_reward=big_reward,
    punishment=0.0,
    random_start=False,
).to(device)

# Base agent
base_agent = BaseAgent(TODO)

# Meta env
meta_env = get_meta_env()

# Meta agent networks
meta_agent = MetaAgent(TODO)

collector = SyncDataCollector(
    env,
    base_policy,
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
    project="toy_ppo",
    name=f"toy_ppo|{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
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
        "left_reward": env.left_reward,
        "right_reward": env.right_reward,
        "down_reward": env.down_reward,
        "up_reward": env.up_reward,
        "n_pos": n_pos,
        "big_reward": env.big_reward,
        "punishment": env.punishment,
    },
)

pbar = tqdm(total=total_frames)

for meta_episode in range(n_meta_episodes):
    for i, base_td in enumerate(
        collector
    ):  # each batch for the base agent is one meta-step
        # Optimization steps for base policy
        for _ in range(num_optim_epochs):
            base_advantage_module(base_td)
            replay_buffer.extend(base_td)
            # advantage_module(td)
            for _ in range(batch_size // sub_batch_size):
                subdata = replay_buffer.sample(sub_batch_size)
                base_loss_vals = base_loss_module(subdata)
                base_loss = (
                    base_loss_vals["loss_objective"]
                    + base_loss_vals["loss_critic"]
                    # + loss_vals["loss_entropy"]
                )
                base_loss.backward()
                base_grad_norm = nn.utils.clip_grad_norm_(
                    base_loss_module.parameters(), max_grad_norm
                )
                base_optim.step()
                base_optim.zero_grad()
        # Skip the first meta step since we don't have a previous meta step
        if not (meta_episode == 0 and i == 0):
            # Assume that we have a `prev_meta_td` with all neccessary information except reward, which we get now
            meta_td["next", "reward"] = base_td["next", "reward"].sum()
            if i == len(collector) - 1:  # end of meta episode
                meta_td["next", "done"] = True
            else:
                meta_td["next", "done"] = False
            # Update meta policy using meta loss
            print(f"before meta GAE: {meta_td=}")
            meta_advantage_module(meta_td)
            meta_loss_vals = meta_loss_module(meta_td)
            meta_loss = meta_loss_vals["loss_objective"] + meta_loss_vals["loss_critic"]
            meta_loss.backward()
            meta_grad_norm = nn.utils.clip_grad_norm_(
                meta_loss_module.parameters(), max_grad_norm
            )
            meta_optim.step()
            meta_optim.zero_grad()

        # The current meta state is mean & std. of the reward distribution from the **previous** batch
        meta_td = TensorDict(
            {
                "meta_state": torch.tensor(
                    [
                        base_td["next", "reward"].mean(),
                        base_td["next", "reward"].std(),
                    ]
                ),
            },
            batch_size=(),
        )
        print(f"before policy: {meta_td=}")
        # Get meta action, a probability to set the constraint
        meta_td = meta_policy(meta_td)
        print(f"after policy: {meta_td=}")
        # Apply meta action
        meta_action_prob = meta_td["probs"]
        meta_action = torch.bernoulli(meta_action_prob).bool().item()
        env.set_constraint_state(meta_action)

        # Update logs
        pbar.update(td.numel())
        if not (meta_episode == 0 and i == 0):
            wandb.log(
                {
                    "base reward": base_td["next", "reward"].float().mean().item(),
                    "base loss_objective": base_loss_vals["loss_objective"].item(),
                    "base loss_critic": base_loss_vals["loss_critic"].item(),
                    "base loss": base_loss.item(),
                    "base state distribution": wandb.Histogram(
                        base_td["pos"].cpu().numpy()
                    ),
                    "base reward distribution": wandb.Histogram(
                        base_td["next", "reward"].cpu().numpy()
                    ),
                    # "state distribution": wandb.Histogram(np.random.randn(10, 10)),
                    "base grad_norm": base_grad_norm.item(),
                    "meta action": meta_action,
                    "meta action prob": meta_action_prob.item(),
                    "meta reward": meta_td["next", "reward"].item(),
                }
            )
        # wandb.Histogram(pos=td["pos"].cpu())
        # if i % eval_every_n_epoch == 0:
        #     with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        #         # execute a rollout with the (greedy) policy and calculate return
        #         eval_rollout = env.rollout(max_rollout_steps, base_policy)
        #         wandb.log(
        #             {
        #                 "eval return": eval_rollout["next", "reward"].sum().item(),
        #                 "eval length": eval_rollout["step_count"].max().item(),
        #             }
        #         )
        # prev_base_td = base_td.clone()

# After training, do a single rollout and print all states and actions
# with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
#     td = env.rollout(max_rollout_steps, base_policy)
#     for i in range(len(td)):
#         print(
#             f"Step {i}: pos={td['pos'][i].item()}, action={td['action'][i].item()}, reward={td['next','reward'][i].item()} done={td['next','done'][i].item()}"
#         )
