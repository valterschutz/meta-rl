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

from env import get_base_env
from agents import BaseAgent
from utils import log, print_base_rollout

device = torch.device("cpu")
n_actions = 4

optimal_return = 0.2  # Optimal return using slow path
gap = 0.1  # How much worse the fast path is
big_reward = 10.0
n_states = 30
init_constraints = False  # Whether to start with constraints enabled
constraints_when = 0.3  # Whether to enable constraints at some fraction of the way
# TODO: use discounting?

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


agent = BaseAgent(
    state_spec=env.state_spec,
    action_spec=env.action_spec,
    num_optim_epochs=10,
    buffer_size=100,
    sub_batch_size=100,
    device=device,
    max_grad_norm=1,
    lr=1e-2,
)

collector = SyncDataCollector(
    env,
    agent.policy,
    frames_per_batch=agent.buffer_size,
    total_frames=10_000,
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
        "constraints_when": constraints_when,
        "lr": agent.lr,
        "num_optim_epochs": agent.num_optim_epochs,
    },
)

pbar = tqdm(total=collector.total_frames)


n_batches = collector.total_frames // collector.frames_per_batch
for i, td in enumerate(collector):
    if constraints_when is not None and i / n_batches >= constraints_when:
        env.set_constraint_state(True)
    losses, max_grad_norm = agent.process_batch(td)

    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        # Evaluation. Always have constraints in the evaluation
        prev_constraints = env.constraints_enabled
        env.set_constraint_state(True)
        eval_td = env.rollout(10 * env.n_states, agent.policy)
        baseline_td = env.rollout(10 * env.n_states, baseline_policy)
        env.set_constraint_state(prev_constraints)
    # print_rollout(eval_td)
    # print(f"max_grad_norm: {max_grad_norm}")
    wandb.log(
        {
            "state distribution": wandb.Histogram(td["state"]),
            "reward distribution": wandb.Histogram(td["next", "reward"]),
            "loss_objective": losses["loss_objective"].item(),
            "loss_critic": losses["loss_critic"].item(),
            "loss_entropy": losses["loss_entropy"].item(),
            "max_grad_norm": max_grad_norm,
            "eval return": eval_td["next", "reward"].sum().item(),
            "baseline return": baseline_td["next", "reward"].sum().item(),
            "eval state distribution": wandb.Histogram(eval_td["state"]),
            "eval reward distribution": wandb.Histogram(eval_td["next", "reward"]),
            "constraints active": float(env.constraints_enabled),
        }
    )
    pbar.update(td.numel())
