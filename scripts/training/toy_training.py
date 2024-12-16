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


def train(env, agent, collector):
    """
    Train the agent on the given environment using the given collector
    """
    pbar = tqdm(total=collector.total_frames)

    try:
        for i, td in enumerate(collector):
            losses, additional_info = agent.process_batch(td)

            qvalue_network_norm = sum(
                (p**2).sum().item()
                for p in agent.loss_module.qvalue_network_params.parameters()
            )
            policy_network_norm = sum(
                (p**2).sum().item()
                for p in agent.loss_module.actor_network_params.parameters()
            )

            loss_dict = {k: v.item() for k, v in losses.items()}
            wandb.log(
                {
                    **loss_dict,
                    **additional_info,
                    "constraints": agent.use_constraints,
                    "mean normal reward": np.mean(
                        td["next", "normal_reward"].cpu().numpy()
                    ),
                    "mean constraint reward": np.mean(
                        td["next", "constraint_reward"].cpu().numpy()
                    ),
                    "buffer size": len(agent.replay_buffer),
                    "batch number": i,
                    "train state distribution": wandb.Histogram(
                        td["state"].cpu().numpy().argmax(-1)
                    ),
                    "train normal reward distribution": wandb.Histogram(
                        td["next", "normal_reward"].cpu().numpy()
                    ),
                    "train constraint reward distribution": wandb.Histogram(
                        td["next", "constraint_reward"].cpu().numpy()
                    ),
                    "qvalue params norm": qvalue_network_norm,
                    "policy params norm": policy_network_norm,
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
    with open("configs/envs/toy_env.yaml", encoding="UTF-8") as f:
        env_config = yaml.safe_load(f)
    if sys.argv[1] == "SAC":
        agent_type = "SAC"
        with open("configs/agents/toy/toy_sac.yaml", encoding="UTF-8") as f:
            agent_config = yaml.safe_load(f)
    elif sys.argv[1] == "DDPG":
        agent_type = "DDPG"
        with open("configs/agents/toy/toy_ddpg.yaml", encoding="UTF-8") as f:
            agent_config = yaml.safe_load(f)
    else:
        raise ValueError("Invalid agent type. Choose either SAC or DDPG.")
    with open("configs/collectors/toy_collector.yaml", encoding="UTF-8") as f:
        collector_config = yaml.safe_load(f)

    env = get_toy_env(env_config, agent_config["gamma"])

    agent = get_toy_agent(agent_type, agent_config, env)

    collector = SyncDataCollector(
        env,
        agent.policy,
        frames_per_batch=collector_config["batch_size"],
        total_frames=collector_config["total_frames"],
        split_trajs=False,
        device=collector_config["device"],
    )

    wandb.init(
        project="toy-base-train",
        name=f"toy-base-train|{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')}",
        config={
            **{f"env_{k}": v for k, v in env_config.items()},
            **{f"agent_{k}": v for k, v in agent_config.items()},
            **{f"collector_{k}": v for k, v in collector_config.items()},
        },
    )

    train(env, agent, collector)


if __name__ == "__main__":
    main()
