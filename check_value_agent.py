# Verify that value agent is correctly implemented

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

from env import get_base_env
from agents import ValueIterationAgent
from utils import log, print_base_rollout

device = torch.device("cpu")
n_actions = 4

optimal_return = 0.3  # Optimal return using slow path
gap = 0.2  # How much worse the fast path is
big_reward = 10.0
n_states = 10

# Assuming n_pos is even, the below equations should hold
# (n_pos-2)*x + big_reward = optimal_return
x = (optimal_return - big_reward) / (n_states - 2)
# (n_pos-2)/2*y + big_reward = optimal_return - gap
y = (optimal_return - gap - big_reward) * 2 / (n_states - 2)
print(f"x: {x}, y: {y}")

# Base env
env = get_base_env(
    left_reward=x,
    right_reward=x,
    down_reward=y,
    up_reward=y,
    n_states=n_states,
    big_reward=big_reward,
    random_start=False,
    punishment=min(-x, -y) / 2,
    # punishment=0,
    seed=None,
    device="cpu",
    constraints_enabled=False,
).to(device)
check_env_specs(env)

agent = ValueIterationAgent(env, gamma=0.99)
agent.update_values()
print(f"Q-values without constraints:")
print(agent.Q)

# Check optimal behavior without constraints
td = env.rollout(100, agent.policy)
print(f"Optimal rollout without constraints:")
print_base_rollout(td)

env.constraints_enabled = True
agent.update_values()
print(f"Q-values with constraints:")
print(agent.Q)

# Check optimal behavior with constraints
td = env.rollout(100, agent.policy)
print(f"Optimal rollout with constraints:")
print_base_rollout(td)
