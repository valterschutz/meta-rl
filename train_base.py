# TODO:
# - [ ] Make sure that base agent loss converges in each meta episode before applying meta actions
# - [ ]

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
import numpy as np

from env import BaseEnv
from agents import BaseAgent, slow_policy, fast_policy, ValueIterationAgent
from utils import log, print_base_rollout, calc_return

device = torch.device("cpu")
n_actions = 4

# init_constraints = False  # Whether to start with constraints enabled
# constraints_when = 0.3  # Whether to enable constraints at some fraction of the way

return_x = 0.2  # Optimal return using slow path
return_y = 0.1  # Return for using fast path
big_reward = 10.0
n_states = 20
gamma = 0.9
rollout_timeout = 10 * n_states
no_weight_fraction = 0.48  # How far into the training should the constraints be off
full_weight_fraction = (
    0.52  # How far into the training should the constraints be fully active
)

# Assuming n_pos is even, calculate x and y
x, y = BaseEnv.calculate_xy(n_states, return_x, return_y, big_reward, gamma)
print(f"x: {x}, y: {y}")

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
    constraints_enabled=True,
).to(device)
check_env_specs(env)
# env.set_constraint_weight(0.33)
# env.set_constraint_state()


# A couple of baseline agents
value_agent = ValueIterationAgent(env, gamma=gamma)
value_agent.update_values()
# print(f"Q-values: {value_agent.Q}")
# fail
# also slow_policy and fast_policy

agent = BaseAgent(
    state_spec=env.state_spec,
    action_spec=env.action_spec,
    num_optim_epochs=10,
    buffer_size=20,
    sub_batch_size=20,
    device=device,
    max_grad_norm=1,
    lr=1e-2,
    gamma=gamma,
    lmbda=0.5,
)

collector = SyncDataCollector(
    env,
    agent.policy,
    frames_per_batch=agent.buffer_size,
    total_frames=1_000,
    split_trajs=False,
    device="cpu",
)

wandb.login()
wandb.init(
    project="base_toy",
    name=f"base_toy|{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    config={
        "batch_size": collector.frames_per_batch,
        "return_x": return_x,
        "return_y": return_y,
        "gamma": gamma,
        "total_frames": collector.total_frames,
        "left_reward": env.left_reward,
        "right_reward": env.right_reward,
        "down_reward": env.down_reward,
        "up_reward": env.up_reward,
        "n_states": env.n_states,
        "big_reward": env.big_reward,
        "punishment": env.punishment,
        "full_weight_fraction": full_weight_fraction,
        # "init_constraints": init_constraints,
        # "constraints_when": constraints_when,
        "lr": agent.lr,
        "num_optim_epochs": agent.num_optim_epochs,
    },
)

pbar = tqdm(total=collector.total_frames)


n_batches = collector.total_frames // collector.frames_per_batch
for i, td in enumerate(collector):
    # constraint_weight = np.clip(
    #     (i - no_weight_fraction * n_batches)
    #     / ((full_weight_fraction - no_weight_fraction) * n_batches),
    #     0,
    #     1,
    # ).item()
    constraint_weight = 1
    env.set_constraint_weight(constraint_weight)
    value_agent.update_values()

    losses, max_grad_norm = agent.process_batch(td)

    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        # Evaluation. Always have constraints in the evaluation
        # prev_constraints = env.constraints_enabled
        # env.set_constraint_state(True)
        eval_td = env.rollout(rollout_timeout, agent.policy)
        # baseline_td = env.rollout(rollout_timeout, baseline_policy)
        # env.set_constraint_state(prev_constraints)
    agent_return = calc_return(eval_td, gamma)
    # Slow policy benchmark
    slow_td = env.rollout(rollout_timeout, slow_policy)
    slow_return = calc_return(slow_td, gamma)
    # Fast policy benchmark
    fast_td = env.rollout(rollout_timeout, fast_policy)
    fast_return = calc_return(fast_td, gamma)
    # Optimal policy benchmark
    optimal_td = env.rollout(rollout_timeout, value_agent.policy)
    optimal_return = calc_return(optimal_td, gamma)
    wandb.log(
        {
            "constraint_weight": constraint_weight,
            # "constraint_weight": 1,
            "train state distribution": wandb.Histogram(td["state"]),
            "train reward distribution": wandb.Histogram(td["next", "reward"]),
            "loss_objective": losses["loss_objective"].item(),
            "loss_critic": losses["loss_critic"].item(),
            "loss_entropy": losses["loss_entropy"].item(),
            "max_grad_norm": max_grad_norm,
            "eval return": agent_return,
            "eval optimal return": optimal_return,
            "eval slow return": slow_return,
            "eval fast return": fast_return,
            "eval state distribution": wandb.Histogram(eval_td["state"]),
            "eval reward distribution": wandb.Histogram(eval_td["next", "reward"]),
            "constraints active": float(env.constraints_enabled),
        }
    )
    pbar.update(td.numel())
