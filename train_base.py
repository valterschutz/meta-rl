import json
import sys
from datetime import datetime, timezone

import numpy as np
import torch
from torchrl.envs.utils import (
    ExplorationType,
    set_exploration_type,
)
from tqdm import tqdm

import wandb
from agents import ValueIterationAgent, fast_policy, slow_policy
from base import get_base_from_config, print_base_rollout
from utils import calc_return, DictWrapper
import yaml


def train_base(config, interactive=None):
    """
    If using this function with sweeps, let interactive be None.
    Otherwise, it should be a dict with the keys
    {"no_weight_fraction", "full_weight_fraction"}.
    """

    env, agent, collector_fn = get_base_from_config(config)
    collector = collector_fn()

    # A couple of baseline agents to use in the verbose case
    if interactive:
        value_agent = ValueIterationAgent(env, gamma=config.gamma)
        value_agent.update_values()

    # Calculate optimal return using value iteration
    value_agent = ValueIterationAgent(env, gamma=config.gamma)
    value_agent.update_values()
    optimal_td = env.rollout(config.rollout_timeout, value_agent.policy)
    optimal_return = calc_return(optimal_td["next", "reward"].flatten(), config.gamma)

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
        agent_return = calc_return(eval_td["next", "reward"].flatten(), config.gamma)
        agent_true_return = calc_return(eval_td["true_reward"].flatten(), config.gamma)

        # Additional benchmarks in the interactive case, and also logging
        if interactive:
            # Slow policy benchmark
            slow_td = env.rollout(config.rollout_timeout, slow_policy)
            # print_base_rollout(slow_td, config.gamma)
            slow_return = calc_return(slow_td["next", "reward"].flatten(), config.gamma)
            slow_true_return = calc_return(
                slow_td["true_reward"].flatten(), config.gamma
            )
            # Fast policy benchmark
            fast_td = env.rollout(config.rollout_timeout, fast_policy)
            fast_return = calc_return(fast_td["next", "reward"].flatten(), config.gamma)
            fast_true_return = calc_return(
                fast_td["true_reward"].flatten(), config.gamma
            )
            # Optimal policy benchmark
            optimal_td = env.rollout(config.rollout_timeout, value_agent.policy)
            optimal_return = calc_return(
                optimal_td["next", "reward"].flatten(), config.gamma
            )
            optimal_true_return = calc_return(
                optimal_td["true_reward"].flatten(), config.gamma
            )
            wandb.log(
                {
                    "constraint_weight": constraint_weight,
                    "train state distribution": wandb.Histogram(td["state"]),
                    "train reward distribution": wandb.Histogram(td["next", "reward"]),
                    "loss_critic": losses["loss_critic"].item(),
                    "loss_entropy": losses["loss_entropy"].item(),
                    "loss_objective": losses["loss_objective"].item(),
                    "max_grad_norm": max_grad_norm,
                    "eval return": agent_return,
                    "eval true return": agent_true_return,
                    "eval optimal return": optimal_return,
                    "eval slow return": slow_return,
                    "eval fast return": fast_return,
                    "eval optimal true return": optimal_true_return,
                    "eval slow true return": slow_true_return,
                    "eval fast true return": fast_true_return,
                    "eval state distribution": wandb.Histogram(eval_td["state"]),
                    "eval reward distribution": wandb.Histogram(
                        eval_td["next", "reward"]
                    ),
                }
            )
        else:  # Not interactive, calculate metrics for hyperparameter sweeps
            batch_dissimilarity = (optimal_return - agent_return) / optimal_return
            return_dissimilarity += batch_dissimilarity
        pbar.update(td.numel())
    return return_dissimilarity / n_batches


if __name__ == "__main__":
    # Treat first argument of program as path to YAML file with config
    with open(sys.argv[1], encoding="UTF-8") as f:
        config = yaml.safe_load(f)

    wandb.init(
        project="toy-base-train",
        name=f"toy-base-train|{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')}",
        config=config,
    )
    train_base(
        DictWrapper(config),
        interactive={
            "no_weight_fraction": config["no_weight_fraction"],
            "full_weight_fraction": config["full_weight_fraction"],
        },
    )
