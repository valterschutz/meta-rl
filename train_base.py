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
from utils import log, print_base_rollout, calc_return, DictWrapper

# device = torch.device("cpu")
# n_actions = 4

# init_constraints = False  # Whether to start with constraints enabled
# constraints_when = 0.3  # Whether to enable constraints at some fraction of the way

# return_x = 0.2  # Optimal return using slow path
# return_y = 0.1  # Return for using fast path
# big_reward = 10.0
# n_states = 20
# gamma = 0.9
# rollout_timeout = 10 * n_states
# no_weight_fraction = 0.48  # How far into the training should the constraints be off
# full_weight_fraction = (
#     0.52  # How far into the training should the constraints be fully active
# )


def train_base(config, interactive=None):
    """
    If using this function with sweeps, let interactive be None.
    Otherwise, it should be a dict with the keys
    {"no_weight_fraction", "full_weight_fraction"}.
    """
    # If batch_size > total_frames, set batch_size to total_frames
    if config.batch_size > config.total_frames:
        batch_size = config.total_frames
    else:
        batch_size = config.batch_size
    # Assuming n_pos is even, calculate x and y
    x, y = BaseEnv.calculate_xy(
        config.n_states,
        config.return_x,
        config.return_y,
        config.big_reward,
        config.gamma,
    )
    # Base env
    env = BaseEnv.get_base_env(
        left_reward=x,
        right_reward=x,
        down_reward=y,
        up_reward=y,
        n_states=config.n_states,
        big_reward=config.big_reward,
        random_start=False,
        punishment=config.punishment,
        seed=None,
        device="cpu",
        constraints_enabled=config.constraints_enabled,
    ).to(config.device)
    check_env_specs(env)

    # A couple of baseline agents to use in the verbose case
    if interactive:
        value_agent = ValueIterationAgent(env, gamma=config.gamma)
        value_agent.update_values()

    agent = BaseAgent(
        state_spec=env.state_spec,
        action_spec=env.action_spec,
        num_optim_epochs=config.num_optim_epochs,
        buffer_size=batch_size,
        sub_batch_size=batch_size,
        device=config.device,
        max_grad_norm=config.max_grad_norm,
        lr=config.lr,
        gamma=config.gamma,
        lmbda=config.lmbda,
    )

    collector = SyncDataCollector(
        env,
        agent.policy,
        frames_per_batch=agent.buffer_size,
        total_frames=config.total_frames,
        split_trajs=False,
        device=config.device,
    )

    # Calculate optimal return using value iteration
    value_agent = ValueIterationAgent(env, gamma=config.gamma)
    value_agent.update_values()
    optimal_td = env.rollout(config.rollout_timeout, value_agent.policy)
    optimal_return = calc_return(optimal_td, config.gamma)

    pbar = tqdm(total=collector.total_frames)

    n_batches = collector.total_frames // collector.frames_per_batch
    return_dissimilarity = 0.0  # How close we are to the optimal return
    for i, td in enumerate(collector):
        if interactive:
            constraint_weight = np.clip(
                (i - interactive["no_weight_fraction"] * n_batches)
                / (
                    (
                        interactive["full_weight_fraction"]
                        - interactive["no_weight_fraction"]
                    )
                    * n_batches
                ),
                0,
                1,
            ).item()
            env.set_constraint_weight(constraint_weight)
            value_agent.update_values()

        losses, max_grad_norm = agent.process_batch(td)

        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            eval_td = env.rollout(config.rollout_timeout, agent.policy)
        agent_return = calc_return(eval_td, config.gamma)
        batch_dissimilarity = (optimal_return - agent_return) / optimal_return
        return_dissimilarity += batch_dissimilarity

        # Additional benchmarks in the interactive case, and also logging
        if interactive:
            # Slow policy benchmark
            slow_td = env.rollout(config.rollout_timeout, slow_policy)
            slow_return = calc_return(slow_td, config.gamma)
            # Fast policy benchmark
            fast_td = env.rollout(config.rollout_timeout, fast_policy)
            fast_return = calc_return(fast_td, config.gamma)
            # Optimal policy benchmark
            optimal_td = env.rollout(config.rollout_timeout, value_agent.policy)
            optimal_return = calc_return(optimal_td, config.gamma)
            print(
                "logging stuff: ",
                optimal_return,
                agent_return,
                slow_return,
                fast_return,
            )
            wandb.log(
                {
                    "constraint_weight": constraint_weight,
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
                    "eval reward distribution": wandb.Histogram(
                        eval_td["next", "reward"]
                    ),
                }
            )
        pbar.update(td.numel())
    return return_dissimilarity / n_batches


if __name__ == "__main__":
    config = {
        "n_states": 20,
        "return_x": 0.2,
        "return_y": 0.1,
        "big_reward": 10,
        "gamma": 0.99,
        "punishment": 0.0,
        "device": "cpu",
        "num_optim_epochs": 6,
        "batch_size": 100,
        "max_grad_norm": 2.42,
        "lmbda": 0.765,
        "total_frames": 1000,
        "rollout_timeout": 200,
        "lr": 0.00034,
        "constraints_enabled": True,
    }
    wandb.init(
        project="base-train",
        name=f"base-train|{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        config=config,
    )
    train_base(
        DictWrapper(config),
        interactive={"no_weight_fraction": 0.99, "full_weight_fraction": 1.0},
    )
