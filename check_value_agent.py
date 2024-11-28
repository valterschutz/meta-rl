# Verify that value agent is correctly implemented, CHECK
# Interesting behavior with n_states=30: with discounting, the optimal policy is actually to take the slow path in the beginning,
# and switch to the fast path in the end

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

from env import BaseEnv
from agents import ValueIterationAgent, slow_policy, fast_policy
from utils import log, print_base_rollout

device = torch.device("cpu")
n_actions = 4

return_x = 0.2  # Optimal return using slow path
return_y = 0.1  # Return for using fast path
big_reward = 10.0
n_states = 20
gamma = 0.99

# Assuming n_pos is even, calculate x and y
x, y = BaseEnv.calculate_xy(n_states, return_x, return_y, big_reward, gamma)

# Base env
env = BaseEnv.get_base_env(
    left_reward=x,
    right_reward=x,
    down_reward=y,
    up_reward=y,
    n_states=n_states,
    big_reward=big_reward,
    random_start=False,
    punishment=0,
    seed=None,
    device="cpu",
    constraints_enabled=False,
).to(device)
check_env_specs(env)

print(f"Rollout with slow policy, without constraints:")
td = env.rollout(100, slow_policy)
print_base_rollout(td, gamma)
print(f"Rollout with fast policy, without constraints:")
td = env.rollout(100, fast_policy)
print_base_rollout(td, gamma)

agent = ValueIterationAgent(env, gamma=gamma)
agent.update_values()
print(f"Q-values without constraints:")
print(agent.Q)
# Check optimal behavior without constraints
td = env.rollout(100, agent.policy)
print(f"Optimal rollout without constraints:")
print_base_rollout(td, gamma)

env.set_constraint_state(True)

print(f"Rollout with slow policy, with constraints:")
td = env.rollout(100, slow_policy)
print_base_rollout(td, gamma)
print(f"Rollout with fast policy, with constraints:")
td = env.rollout(100, fast_policy)
print_base_rollout(td, gamma)

agent.update_values()
print(f"Q-values with constraints:")
print(agent.Q)
# Check optimal behavior with constraints
td = env.rollout(100, agent.policy)
print(f"Optimal rollout with constraints:")
print_base_rollout(td, gamma)
