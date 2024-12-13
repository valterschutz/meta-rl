import os
import sys
from datetime import datetime, timezone

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from pathlib import Path

import numpy as np
import yaml
from torchrl.collectors import SyncDataCollector
from torchrl.envs.transforms import Compose, Reward2GoTransform
from tqdm import tqdm

import wandb
from src.agents.cartpole_agents import get_cartpole_agent
from src.envs.cartpole_env import get_cartpole_env


def train(env, agent, collector):
    """
    Train the agent on the given environment using the given collector
    """
    pbar = tqdm(total=collector.total_frames)

    try:
        for i, td in enumerate(collector):
            losses, max_grad_norm = agent.process_batch(td)

            loss_dict = {k: v.item() for k, v in losses.items()}
            wandb.log(
                {
                    **loss_dict,
                    "mean reward": np.mean(td["next", "reward"].cpu().numpy()),
                    "max_grad_norm": max_grad_norm,
                    "mean reward_to_go": np.mean(td["reward_to_go"].cpu().numpy()),
                    "mean constraint_to_go": np.mean(
                        td["constraint_to_go"].cpu().numpy()
                    ),
                    # "Q-value network norm": torch.norm(
                    #     torch.stack(
                    #         [
                    #             torch.norm(p)
                    #             for p in agent.loss_module.value_network_params.parameters()
                    #         ]
                    #     )
                    # ).item(),
                    "buffer size": len(agent.replay_buffer),
                    "batch number": i,
                }
            )
            pbar.update(td.numel())
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    pbar.close()

    # Save model weights
    print("Saving policy...")
    p = Path("models/cartpole/DDPG/")
    # Create a directory in `p` with the name of the datetime
    p = p / datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    # Create the directory if not already present
    p.mkdir(parents=True, exist_ok=True)
    policy_path = p / "policy.pth"
    torch.save(agent.policy_module, policy_path)


def main():
    # Configuration only related to how environment reacts
    with open("configs/envs/cartpole_env.yaml", encoding="UTF-8") as f:
        env_config = yaml.safe_load(f)

    env = get_cartpole_env(env_config)

    # Load an agent, depending on which algorithm is chosen ("SAC" or "DDPG")
    agent, agent_config = get_cartpole_agent(sys.argv[1], env.action_spec)
    t = Compose(
        Reward2GoTransform(
            gamma=agent_config["gamma"],
            in_keys=[("next", "reward")],
            out_keys=["reward_to_go"],
        ),
        Reward2GoTransform(
            gamma=agent_config["gamma"],
            in_keys=[("constraint")],
            out_keys=["constraint_to_go"],
        ),
    )

    # Configuration for the collector
    with open("configs/collectors/cartpole_collector.yaml", encoding="UTF-8") as f:
        collector_config = yaml.safe_load(f)

    collector = SyncDataCollector(
        env,
        agent.policy,
        frames_per_batch=collector_config["batch_size"],
        total_frames=collector_config["total_frames"],
        split_trajs=False,
        device=collector_config["device"],
        postproc=t.inv,
    )

    wandb.init(
        project="cartpole-base-train",
        name=f"cartpole-base-train|{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')}",
        config={
            **{f"env_{k}": v for k, v in env_config.items()},
            **{f"agent_{k}": v for k, v in agent_config.items()},
            **{f"collector_{k}": v for k, v in collector_config.items()},
        },
    )

    train(env, agent, collector)


if __name__ == "__main__":
    main()
