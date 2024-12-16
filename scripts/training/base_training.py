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
from src.agents import get_toy_agent
from src.envs.toy_env import get_toy_env, ToyEnv
from src.loss_modules import (
    get_discrete_sac_loss_module,
    get_continuous_sac_loss_module,
)

import argparse

import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from tensordict.nn import InteractionType, TensorDictModule
from torchrl.modules import (
    ProbabilisticActor,
    ValueOperator,
    NormalParamExtractor,
    TruncatedNormal,
)
from torchrl.objectives import DiscreteSACLoss, ValueEstimators, SACLoss
from torch.distributions import OneHotCategorical
from torchrl.envs import DMControlEnv
from torchrl.envs.utils import check_env_specs
from torchrl.envs.transforms import (
    CatTensors,
    Compose,
    DoubleToFloat,
    RenameTransform,
    StepCounter,
    TransformedEnv,
)
from torchrl.data.replay_buffers import ReplayBuffer
from tensordict import TensorDict
from torchrl.objectives.utils import SoftUpdate
from src.agents import OffpolicyAgent
from src.loss_modules import (
    get_discrete_sac_loss_module,
    get_continuous_sac_loss_module,
)


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
    print("Saving agent...")
    agent.save("models/{env_type}/{agent_type}/")
    # p = Path(f"models/toy/{agent.__class__.__name__}/")
    # # Create a directory in `p` with the name of the datetime
    # p = p / datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    # # Create the directory if not already present
    # p.mkdir(parents=True, exist_ok=True)
    # policy_path = p / "policy.pth"
    # torch.save(agent.policy_module, policy_path)


def get_env(env_type, env_config, gamma):
    if env_type == "toy":
        x, y = ToyEnv.calculate_xy(
            env_config["n_states"],
            env_config["return_x"],
            env_config["return_y"],
            env_config["big_reward"],
            gamma,
        )
        env = ToyEnv(
            left_reward=x,
            right_reward=x,
            down_reward=y,
            up_reward=y,
            n_states=env_config["n_states"],
            big_reward=env_config["big_reward"],
            random_start=False,
            constraints_active=env_config["constraints_active"],
            seed=None,
            device=env_config["device"],
        ).to(env_config["device"])
        check_env_specs(env)
    elif env_type == "cartpole":

        def constraint_transform(td):
            # Constraint reward:
            td["constraint"] = (td["velocity"] ** 2).sum(-1, keepdim=True)
            return td

        env = TransformedEnv(
            DMControlEnv(
                "cartpole",
                "swingup",
                device=env_config["device"],
                # from_pixels=from_pixels,
            ),
            Compose(
                DoubleToFloat(),
                CatTensors(
                    in_keys=["position", "velocity"], out_key="state", del_keys=False
                ),
                RenameTransform(
                    in_keys=["reward"], out_keys=["normal_reward"], create_copy=True
                ),
                StepCounter(),
                constraint_transform,
            ),
        )

    return env


def get_agent(agent_type, agent_config, env_type, env):
    if agent_type == "SAC":
        if env_type == "toy":
            loss_module = get_discrete_sac_loss_module(
                n_states=env.n_states,
                action_spec=env.action_spec,
                target_entropy=agent_config["target_entropy"],
                gamma=agent_config["gamma"],
            )
        elif env_type == "cartpole":
            loss_module = get_continuous_sac_loss_module(
                n_states=5,
                n_actions=1,
                action_spec=env.action_spec,
                target_entropy=agent_config["target_entropy"],
                gamma=agent_config["gamma"],
            )
        else:
            raise NotImplementedError(f"Environment type {type(env)} not implemented.")
        target_updater = SoftUpdate(loss_module, eps=agent_config["target_eps"])
        optims = [torch.optim.Adam(loss_module.parameters(), lr=agent_config["lr"])]
        loss_keys = ["loss_actor", "loss_qvalue", "loss_alpha"]
    else:
        raise NotImplementedError(f"Agent type {agent_type} not implemented.")

    return OffpolicyAgent(
        target_updater=target_updater,
        optims=optims,
        loss_keys=loss_keys,
        loss_module=loss_module,
        **agent_config,
    )


def main():
    parser = argparse.ArgumentParser(description="Train a base agent.")
    parser.add_argument(
        "agent_type", choices=["SAC", "DDPG", "TD3"], help="Type of agent to train"
    )
    parser.add_argument(
        "env_type", choices=["toy", "cartpole"], help="Type of environment to train in"
    )
    args = parser.parse_args()
    with open(f"configs/envs/{args.env_type}_env.yaml".lower(), encoding="UTF-8") as f:
        env_config = yaml.safe_load(f)
    with open(
        f"configs/agents/{args.env_type}/{args.env_type}_{args.agent_type}.yaml".lower(),
        encoding="UTF-8",
    ) as f:
        agent_config = yaml.safe_load(f)
    with open(
        f"configs/collectors/{args.env_type}_collector.yaml".lower(), encoding="UTF-8"
    ) as f:
        collector_config = yaml.safe_load(f)

    env = get_env(args.env_type, env_config, agent_config["gamma"])
    agent = get_agent(args.agent_type, agent_config, args.env_type, env)

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
