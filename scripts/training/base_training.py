import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from tensordict.nn.probabilistic import InteractionType
from torchrl.collectors import SyncDataCollector
from torchrl.envs import DMControlEnv
from torchrl.envs.transforms import (CatTensors, Compose, DoubleToFloat,
                                     RenameTransform, StepCounter,
                                     TransformedEnv)
from torchrl.envs.utils import check_env_specs, set_exploration_type
from torchrl.objectives import SoftUpdate
from torchrl.record import CSVLogger, VideoRecorder
from tqdm import tqdm

import wandb

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.agents import OffpolicyTrainer
from src.envs.dm_env import get_cartpole_env, get_pendulum_env, get_reacher_env
from src.envs.toy_env import get_toy_env
from src.loss_modules import (get_continuous_sac_loss_module,
                              get_continuous_td3_loss_module,
                              get_discrete_sac_loss_module)


def train(env, trainer, collector, env_type:str, agent_type:str, pixel_env=None):
    """
    Train the agent on the given environment using the given collector
    """
    pbar = tqdm(total=collector.total_frames)

    try:
        for i, td in enumerate(collector):
            losses, additional_info = trainer.process_batch(td)

            # We want to plot the action distribution in the environment
            # If the actions are one_hot encoded, we can plot the argmax of the action
            action_distribution = wandb.Histogram(td["action"].argmax(dim=-1).cpu()) if env_type == "toy" else wandb.Histogram(td["action"].cpu())

            loss_dict = {k: v.item() for k, v in losses.items()}
            wandb.log(
                {
                    **loss_dict,
                    **additional_info,
                    "batch number": i,
                    "qvalue_network_ss": sum((p**2).mean().item() for p in trainer.loss_module.qvalue_network_params.parameters()),
                    "policy_network_ss": sum((p**2).mean().item() for p in trainer.loss_module.actor_network_params.parameters()),
                    "action distribution": action_distribution,
                }
            )
            pbar.update(td.numel())
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    # Log a video of final performance if pixel_env is provided
    if pixel_env is not None:
        print("Recording video...")
        dt = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        exp_name = f"{env_type}|{agent_type}_{dt}"
        logger = CSVLogger(exp_name=exp_name, log_dir="logs", video_format="pt")
        recorder = VideoRecorder(logger, tag="temp")
        record_env = TransformedEnv(pixel_env, recorder)
        with torch.no_grad(), set_exploration_type(InteractionType.DETERMINISTIC):
            rollout = record_env.rollout(max_steps=sys.maxsize, policy=trainer.loss_module.actor_network)
        recorder.dump() # Saves video as a .pt file at `csv`
        # Now load the file as a tensor
        video = torch.load(Path(logger.log_dir) / exp_name / "videos" / "temp_0.pt").numpy()
        wandb.log({"video": wandb.Video(video)})
    pbar.close()

    # Save model weights
    print("Saving agent...")
    save_path = Path(f"models/{env_type}/{agent_type}")
    # Create necessary directories if not already present
    save_path.mkdir(parents=True, exist_ok=True)
    trainer.save(save_path)


def get_env(env_type, env_config, gamma):
    if env_type == "toy":
        env, pixel_env = get_toy_env(env_config, gamma)
    elif env_type == "cartpole":
        env, pixel_env = get_cartpole_env(env_config)
    elif env_type == "pendulum":
        env, pixel_env = get_pendulum_env(env_config)
    elif env_type == "reacher":
        env, pixel_env = get_reacher_env(env_config)
    else:
        raise NotImplementedError(f"Environment type {env_type} not implemented.")

    return env, pixel_env


def get_trainer_and_policy(agent_type, agent_config, env_type, env):
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
        elif env_type == "pendulum":
            loss_module = get_continuous_sac_loss_module(
                n_states=3,
                n_actions=1,
                action_spec=env.action_spec,
                target_entropy=agent_config["target_entropy"],
                gamma=agent_config["gamma"],
            )
        elif env_type == "reacher":
            loss_module = get_continuous_sac_loss_module(
                n_states=4,
                n_actions=2,
                action_spec=env.action_spec,
                target_entropy=agent_config["target_entropy"],
                gamma=agent_config["gamma"],
                action_low=-1,
                action_high=1,
            )
        else:
            raise NotImplementedError(f"SAC not implemented for environment {type(env)}")
        loss_keys = ["loss_actor", "loss_qvalue", "loss_alpha"]
        loss_module = loss_module.to(agent_config["device"])
        target_updater = SoftUpdate(loss_module, eps=agent_config["target_eps"])
        optims = [torch.optim.Adam(loss_module.parameters(), lr=agent_config["lr"])]
    elif agent_type == "TD3":
        if env_type == "pendulum":
            loss_module = get_continuous_td3_loss_module(
                n_states=3,
                n_actions=1,
                action_spec=env.action_spec,
                gamma=agent_config["gamma"],
            )
        elif env_type == "cartpole":
            loss_module = get_continuous_td3_loss_module(
                n_states=5,
                n_actions=1,
                action_spec=env.action_spec,
                gamma=agent_config["gamma"],
            )
        else:
            raise NotImplementedError(f"TD3 not implemented for environment {type(env)}")
        loss_keys = ["loss_actor", "loss_qvalue"]
        loss_module = loss_module.to(agent_config["device"])
        target_updater = SoftUpdate(loss_module, eps=agent_config["target_eps"])
        optims = [torch.optim.Adam(loss_module.parameters(), lr=agent_config["lr"])]
    else:
        raise NotImplementedError(f"Agent type {agent_type} not implemented.")

    return OffpolicyTrainer(
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
        "env_type", choices=["toy", "cartpole", "pendulum", "reacher"], help="Type of environment to train in"
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

    env, pixel_env = get_env(args.env_type, env_config, agent_config["gamma"])
    trainer = get_trainer_and_policy(
        args.agent_type, agent_config, args.env_type, env
    )

    collector = SyncDataCollector(
        env,
        trainer.loss_module.actor_network,
        frames_per_batch=collector_config["batch_size"],
        total_frames=collector_config["total_frames"],
        split_trajs=False,
        device=collector_config["device"],
    )

    wandb.init(
        project=f"{args.env_type}-{args.agent_type}-base-train",
        name=f"{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')}",
        config={
            **{f"env_{k}": v for k, v in env_config.items()},
            **{f"agent_{k}": v for k, v in agent_config.items()},
            **{f"collector_{k}": v for k, v in collector_config.items()},
        },
    )

    train(env, trainer, collector, env_type=args.env_type, agent_type=args.agent_type, pixel_env=pixel_env)


if __name__ == "__main__":
    main()
