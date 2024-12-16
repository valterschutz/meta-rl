import os
import sys
from datetime import datetime, timezone

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from pathlib import Path

import numpy as np
import yaml
from torchrl.collectors import SyncDataCollector
from tqdm import tqdm

import wandb
from src.agents.toy_agents import get_toy_agent
from src.envs.toy_env import get_toy_env

import argparse


def train(env, agent, collector):
    """
    Train the agent on the given environment using the given collector
    """
    pbar = tqdm(total=collector.total_frames)

    try:
        for i, td in enumerate(collector):
            losses, additional_info = agent.process_batch(td)

            loss_dict = {k: v.item() for k, v in losses.items()}
            wandb.log(
                {
                    **loss_dict,
                    **additional_info,
                    "batch number": i,
                }
            )
            pbar.update(td.numel())
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    pbar.close()

    # Save model weights
    print("Saving policy...")
    p = Path(f"models/toy/{agent.__class__.__name__}/")
    # Create a directory in `p` with the name of the datetime
    p = p / datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    # Create the directory if not already present
    p.mkdir(parents=True, exist_ok=True)
    policy_path = p / "policy.pth"
    torch.save(agent.policy_module, policy_path)


def main():
    parser = argparse.ArgumentParser(description="Train a base agent.")
    parser.add_argument(
        "agent_type", choices=["SAC", "DDPG", "TD3"], help="Type of agent to train"
    )
    parser.add_argument(
        "env_type", choices=["toy", "cartpole"], help="Type of environment to train in"
    )
    args = parser.parse_args()
    with open(f"configs/envs/{args.env_type}_env.yaml", encoding="UTF-8") as f:
        env_config = yaml.safe_load(f)
    with open(
        "configs/agents/{args.env_type}/{args.env_type}_{args.agent_type}.yaml",
        encoding="UTF-8",
    ) as f:
        agent_config = yaml.safe_load(f)
    with open(
        "configs/collectors/{args.env_type}_collector.yaml", encoding="UTF-8"
    ) as f:
        collector_config = yaml.safe_load(f)

    if args.env_type == "toy":
        env = get_toy_env(env_config, agent_config["gamma"])
        agent = get_toy_agent(agent_type, agent_config, env)
    elif args.env_type == "cartpole":
        env = get_cartpole_env(env_config)
        agent = get_cartpole_agent(agent_type, env.action_spec)

    collector = SyncDataCollector(
        env,
        agent.policy,
        frames_per_batch=collector_config["batch_size"],
        total_frames=collector_config["total_frames"],
        split_trajs=False,
        device=collector_config["device"],
    )

    wandb.init(
        project="{args.env_type}-base-train",
        name=f"{args.env_type}-base-train|{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')}",
        config={
            **{f"env_{k}": v for k, v in env_config.items()},
            **{f"agent_{k}": v for k, v in agent_config.items()},
            **{f"collector_{k}": v for k, v in collector_config.items()},
        },
    )

    train(env, agent, collector)


if __name__ == "__main__":
    main()
