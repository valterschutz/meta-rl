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

from env import get_base_env
from agents import DiscreteACAgent
from utils import log, print_base_rollout

device = torch.device("cpu")
n_actions = 4

optimal_return = 0.2  # Optimal return using slow path
gap = 0.1  # How much worse the fast path is
big_reward = 10.0
n_states = 10
init_constraints = False  # Whether to start with constraints enabled
halfway_constraints = True  # Whether to enable constraints halfway through training

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
    punishment=0.0,
    seed=None,
    device="cpu",
    constraints_enabled=False,
).to(device)
check_env_specs(env)
env.set_constraint_state(init_constraints)


# Baseline agent, always goes right (which is optimal)
def baseline_policy(td):
    td["action"] = 1
    return td


agent = DiscreteACAgent(
    n_states=env.n_states,
    n_actions=n_actions,
    device="cpu",
    w_lr=1e-2,
    theta_lr=1e-3,
    num_optim_epochs=100,
)

collector = SyncDataCollector(
    env,
    agent.exploration_policy,
    frames_per_batch=env.n_states,
    total_frames=200 * env.n_states,
    split_trajs=False,
    device="cpu",
)

wandb.login()
wandb.init(
    project="base_toy",
    name=f"base_toy|{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    config={
        "batch_size": collector.frames_per_batch,
        "optimal_return": optimal_return,
        "gap": gap,
        "total_frames": collector.total_frames,
        "left_reward": env.left_reward,
        "right_reward": env.right_reward,
        "down_reward": env.down_reward,
        "up_reward": env.up_reward,
        "n_states": env.n_states,
        "big_reward": env.big_reward,
        "punishment": env.punishment,
        "init_constraints": init_constraints,
        "halfway_constraints": halfway_constraints,
        "w_lr": agent.w_lr,
        "theta_lr": agent.theta_lr,
        "num_optim_epochs": agent.num_optim_epochs,
    },
)

pbar = tqdm(total=collector.total_frames)


n_batches = collector.total_frames // collector.frames_per_batch
for i, td in enumerate(collector):
    if halfway_constraints and i == n_batches // 2:
        env.set_constraint_state(True)
    td_errors = agent.process_batch(td)

    # Evaluation. Always have constraints in the evaluation
    prev_constraints = env.constraints_enabled
    env.set_constraint_state(True)
    eval_td = env.rollout(10 * env.n_states, agent.explotation_policy)
    baseline_td = env.rollout(10 * env.n_states, baseline_policy)
    env.set_constraint_state(prev_constraints)
    # print_rollout(eval_td)
    # print(f"max_grad_norm: {max_grad_norm}")
    value_dict = {f"value of state {i}": agent.w[i].item() for i in range(env.n_states)}
    wandb.log(
        {
            "state distribution": wandb.Histogram(td["state"]),
            "reward distribution": wandb.Histogram(td["next", "reward"]),
            "mean td_error": sum(td_errors) / len(td_errors),
            "max td_error": max(td_errors),
            "eval return": eval_td["next", "reward"].sum().item(),
            "baseline return": baseline_td["next", "reward"].sum().item(),
            "eval state distribution": wandb.Histogram(eval_td["state"]),
            "eval reward distribution": wandb.Histogram(eval_td["next", "reward"]),
            **value_dict,
        }
    )
    pbar.update(td.numel())
