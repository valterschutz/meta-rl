# TODO:
# - [ ] Make sure that base agent loss converges in each meta episode before applying meta actions

from tqdm import tqdm
from datetime import datetime

from torchrl.envs.utils import (
    check_env_specs,
    step_mdp,
    set_exploration_type,
    ExplorationType,
)
from torchrl.collectors import SyncDataCollector

import torch
import wandb

from env import get_base_env, MetaEnv
from agents import BaseAgent, MetaAgent
from utils import log, print_base_rollout

meta_episodes = 2
meta_steps_per_episode = 100  # TODO: 10 should be enough
device = torch.device("cpu")
n_actions = 4

optimal_return = 0.2  # Optimal return using slow path
gap = 0.1  # How much worse the fast path is
big_reward = 10.0
n_pos = 11  # TODO: should be 20?

# Assuming n_pos is odd, the below equations should hold
# (n_pos-1)*x + big_reward = optimal_return
x = (optimal_return - big_reward) / (n_pos - 1)
# (n_pos-1)/2*y + big_reward = optimal_return - gap
y = (optimal_return - gap - big_reward) * 2 / (n_pos - 1)
print(f"x: {x}, y: {y}")

# Base env
base_env = get_base_env(
    left_reward=x,
    right_reward=x,
    down_reward=y,
    up_reward=y,
    n_pos=n_pos,
    big_reward=big_reward,
    random_start=False,
    punishment=0.0,
    seed=None,
    device="cpu",
    constraints_enabled=False,
).to(device)
check_env_specs(base_env)
base_env.set_constraint_state(True)

# Base agent
base_agent = BaseAgent(
    state_spec=base_env.state_spec,
    action_spec=base_env.action_spec,
    num_optim_epochs=10,
    buffer_size=100,
    sub_batch_size=20,
    device="cpu",
    max_grad_norm=1,
    lr=1e-1,
)

base_collector = SyncDataCollector(
    base_env,
    base_agent.policy,
    frames_per_batch=base_agent.buffer_size,
    total_frames=10_000,
    split_trajs=False,
    device="cpu",
)

wandb.login()
wandb.init(
    project="base_toy",
    name=f"base_toy|{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    config={
        "base_agent.buffer_size": base_agent.buffer_size,
        "base_agent.sub_batch_size": base_agent.sub_batch_size,
        "left_reward": base_env.left_reward,
        "right_reward": base_env.right_reward,
        "down_reward": base_env.down_reward,
        "up_reward": base_env.up_reward,
        "n_pos": base_env.n_pos,
        "big_reward": base_env.big_reward,
        "punishment": base_env.punishment,
    },
)

pbar = tqdm(total=base_collector.total_frames)


n_batches = base_collector.total_frames // base_collector.frames_per_batch
for i, td in enumerate(base_collector):
    if i == n_batches // 2:
        base_env.set_constraint_state(True)
    losses, max_grad_norm = base_agent.process_batch(td)
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        eval_td = base_env.rollout(100, base_agent.policy)
        # print_base_rollout(eval_td)
    wandb.log(
        {
            "state distribution": wandb.Histogram(td["pos"]),
            "reward distribution": wandb.Histogram(td["next", "reward"]),
            "loss_objective": losses["loss_objective"],
            "loss_critic": losses["loss_critic"],
            "loss_entropy": losses["loss_entropy"],
            "max_grad_norm": max_grad_norm,
            "eval return": eval_td["next", "reward"].sum().item(),
            "eval state distribution": wandb.Histogram(eval_td["pos"]),
            "eval reward distribution": wandb.Histogram(eval_td["next", "reward"]),
        }
    )
    pbar.update(td.numel())
