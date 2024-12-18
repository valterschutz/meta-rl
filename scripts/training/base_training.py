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

from src.agents import OffpolicyTrainer, OnpolicyTrainer, toy_fast_policy, toy_slow_policy
from src.envs.dm_env import get_cartpole_env, get_pendulum_env, get_reacher_env
from src.envs.toy_env import get_toy_env
from src.loss_modules import (get_continuous_sac_loss_module,
                              get_continuous_td3_loss_module, get_discrete_ppo_loss_module,
                              get_discrete_sac_loss_module)
from src.utils import calc_return

def save_video(env_type, agent_type, trainer, pixel_env, batch_number):
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
    wandb.log({"video": wandb.Video(video), "batch number": batch_number})


def train(env, trainer, collector, env_type:str, agent_type:str, eval_every_nth_batch:int, gamma:float, pixel_env=None):
    """
    Train the agent on the given environment using the given collector
    """
    pbar = tqdm(total=collector.total_frames)

    trainer.use_constraints = True
    try:
        for i, td in enumerate(collector):
            losses, additional_info = trainer.process_batch(td)
            # collector.update_policy_weights_()

            # We want to plot the action and state distribution in the environment
            # If the actions/states are one_hot encoded, we can plot the argmax of them
            # If actions/states are multi-dimensional, we want one histogram per dimension
            # action_distribution = wandb.Histogram(td["action"].argmax(dim=-1).cpu()) if env_type == "toy" else wandb.Histogram(td["action"].cpu())
            # state_distribution = wandb.Histogram(td["state"].argmax(dim=-1).cpu()) if env_type == "toy" else wandb.Histogram(td["state"].cpu())
            state_distributions = {}
            for j in range(td["state"].shape[1]):
                state_distributions[f"state_{j} distribution"] = wandb.Histogram(td["state"][:, j].cpu())
            action_distributions = {}
            for j in range(td["action"].shape[1]):
                action_distributions[f"action_{j} distribution"] = wandb.Histogram(td["action"].argmax(dim=-1)) if env_type == "toy"  else wandb.Histogram(td["action"][:, j].cpu())

            loss_dict = {k: v.item() for k, v in losses.items()}
            # Log "norm" of networks
            norm_dict = {}
            if "qvalue_network" in trainer.loss_module.__dict__:
                norm_dict["qvalue_network_ss"] = sum((p**2).mean().item() for p in trainer.loss_module.qvalue_network_params.parameters())
            if "policy_network" in trainer.loss_module.__dict__:
                norm_dict["policy_network_ss"] = sum((p**2).mean().item() for p in trainer.loss_module.actor_network_params.parameters())
            if "actor_network" in trainer.loss_module.__dict__:
                norm_dict["actor_network_ss"] = sum((p**2).mean().item() for p in trainer.loss_module.actor_network_params.parameters())
            if "critic_network" in trainer.loss_module.__dict__:
                norm_dict["critic_network_ss"] = sum((p**2).mean().item() for p in trainer.loss_module.critic_network_params.parameters())
            wandb.log(
                {
                    **loss_dict,
                    **additional_info,
                    **norm_dict,
                    **action_distributions,
                    **state_distributions,
                    "batch number": i,
                }
            )

            if i % eval_every_nth_batch == 0:
                with torch.no_grad(), set_exploration_type(InteractionType.DETERMINISTIC):
                    rollout = env.rollout(max_steps=sys.maxsize, policy=trainer.loss_module.actor_network)
                    unconstrained_return = calc_return(rollout["next", "normal_reward"].flatten(), gamma)
                    constrained_return = calc_return(rollout["next", "normal_reward"].flatten()+rollout["next", "constraint_reward"].flatten(), gamma)
                    wandb.log(
                        {
                            "batch number": i,
                            "unconstrained return": unconstrained_return,
                            "constrained return": constrained_return,
                        }
                    )
                    if pixel_env is not None:
                        save_video(env_type, agent_type, trainer, pixel_env, batch_number=i)

                    # Temporary debugging for toy env:
                    if env_type == "toy":
                        slow_rollout = env.rollout(max_steps=sys.maxsize, policy=toy_slow_policy)
                        slow_unconstrained_return = calc_return(slow_rollout["next", "normal_reward"].flatten(), gamma)
                        slow_constrained_return = calc_return(slow_rollout["next", "normal_reward"].flatten()+slow_rollout["next", "constraint_reward"].flatten(), gamma)
                        fast_rollout = env.rollout(max_steps=sys.maxsize, policy=toy_fast_policy)
                        fast_unconstrained_return = calc_return(fast_rollout["next", "normal_reward"].flatten(), gamma)
                        fast_constrained_return = calc_return(fast_rollout["next", "normal_reward"].flatten()+fast_rollout["next", "constraint_reward"].flatten(), gamma)
                        wandb.log(
                            {
                                "batch number": i,
                                "slow unconstrained return": slow_unconstrained_return,
                                "slow constrained return": slow_constrained_return,
                                "fast unconstrained return": fast_unconstrained_return,
                                "fast constrained return": fast_constrained_return,
                            }
                        )


            pbar.update(td.numel())
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    # Log a video of final performance if pixel_env is provided
    if pixel_env is not None:
        print("Recording video...")
        save_video(env_type, agent_type, trainer, pixel_env, batch_number=-1)
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


def get_trainer_and_policy(agent_type, agent_config, env_type, env, collector_config):
    if agent_type == "SAC":
        if env_type == "toy":
            loss_module = get_discrete_sac_loss_module(
                n_states=env.n_states,
                action_spec=env.action_spec,
                gamma=agent_config["gamma"],
            )
            loss_keys = ["loss_actor", "loss_qvalue", "loss_alpha"]
            loss_module = loss_module.to(agent_config["device"])
            target_updater = SoftUpdate(loss_module, eps=agent_config["target_eps"])
            optims = [
                torch.optim.Adam(loss_module.actor_network_params.parameters(), lr=agent_config["actor_lr"]),
                torch.optim.Adam(loss_module.qvalue_network_params.parameters(), lr=agent_config["qvalue_lr"]),
            ]
        elif env_type == "cartpole":
            loss_module = get_continuous_sac_loss_module(
                n_states=5,
                n_actions=1,
                action_spec=env.action_spec,
                gamma=agent_config["gamma"],
            )
            loss_keys = ["loss_actor", "loss_qvalue", "loss_alpha"]
            loss_module = loss_module.to(agent_config["device"])
            target_updater = SoftUpdate(loss_module, eps=agent_config["target_eps"])
            optims = [torch.optim.Adam(loss_module.parameters(), lr=agent_config["lr"])]
        elif env_type == "pendulum":
            loss_module = get_continuous_sac_loss_module(
                n_states=3,
                n_actions=1,
                action_spec=env.action_spec,
                gamma=agent_config["gamma"],
            )
            loss_keys = ["loss_actor", "loss_qvalue", "loss_alpha"]
            loss_module = loss_module.to(agent_config["device"])
            target_updater = SoftUpdate(loss_module, eps=agent_config["target_eps"])
            optims = [torch.optim.Adam(loss_module.parameters(), lr=agent_config["lr"])]
        elif env_type == "reacher":
            loss_module = get_continuous_sac_loss_module(
                n_states=4,
                n_actions=2,
                action_spec=env.action_spec,
                gamma=agent_config["gamma"],
                action_low=-1,
                action_high=1,
            )
            loss_keys = ["loss_actor", "loss_qvalue", "loss_alpha"]
            loss_module = loss_module.to(agent_config["device"])
            target_updater = SoftUpdate(loss_module, eps=agent_config["target_eps"])
            optims = [torch.optim.Adam(loss_module.parameters(), lr=agent_config["lr"])]
        else:
            raise NotImplementedError(f"SAC not implemented for environment {type(env)}")
        return OffpolicyTrainer(
                target_updater=target_updater,
                optims=optims,
                loss_keys=loss_keys,
                loss_module=loss_module,
                **agent_config,
        )
    elif agent_type == "TD3":
        if env_type == "pendulum":
            loss_module = get_continuous_td3_loss_module(
                n_states=3,
                n_actions=1,
                action_spec=env.action_spec,
                gamma=agent_config["gamma"],
            )
            loss_keys = ["loss_actor", "loss_qvalue"]
            loss_module = loss_module.to(agent_config["device"])
            target_updater = SoftUpdate(loss_module, eps=agent_config["target_eps"])
            optims = [torch.optim.Adam(loss_module.parameters(), lr=agent_config["lr"])]
        elif env_type == "cartpole":
            loss_module = get_continuous_td3_loss_module(
                n_states=5,
                n_actions=1,
                action_spec=env.action_spec,
                gamma=agent_config["gamma"],
            )
            loss_keys = ["loss_actor", "loss_qvalue"]
            loss_module = loss_module.to(agent_config["device"])
            target_updater = SoftUpdate(loss_module, eps=agent_config["target_eps"])
            optims = [torch.optim.Adam(loss_module.parameters(), lr=agent_config["lr"])]
        else:
            raise NotImplementedError(f"TD3 not implemented for environment {type(env)}")
        return OffpolicyTrainer(
                target_updater=target_updater,
                optims=optims,
                loss_keys=loss_keys,
                loss_module=loss_module,
                **agent_config,
        )
    elif agent_type == "PPO":
        if env_type == "toy":
            loss_module = get_discrete_ppo_loss_module(
                n_states=env.n_states,
                action_spec=env.action_spec,
                gamma=agent_config["gamma"],
            )
            loss_keys = ["loss_objective", "loss_critic", "loss_entropy"]
            loss_module = loss_module.to(agent_config["device"])
            optims = [torch.optim.Adam(loss_module.parameters(), lr=agent_config["lr"])]
        else:
            raise NotImplementedError(f"PPO not implemented for environment {type(env)}")
        return OnpolicyTrainer(
                buffer_size=collector_config["batch_size"],
                optims=optims,
                loss_keys=loss_keys,
                loss_module=loss_module,
                **agent_config,
        )
    else:
        raise NotImplementedError(f"Agent type {agent_type} not implemented.")



def main():
    parser = argparse.ArgumentParser(description="Train a base agent.")
    parser.add_argument(
        "agent_type", choices=["SAC", "DDPG", "TD3", "PPO"], help="Type of agent to train"
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
        args.agent_type, agent_config, args.env_type, env, collector_config
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

    train(env, trainer, collector, env_type=args.env_type, agent_type=args.agent_type, pixel_env=pixel_env, eval_every_nth_batch=collector_config["eval_every_nth_batch"], gamma=agent_config["gamma"])


if __name__ == "__main__":
    main()
