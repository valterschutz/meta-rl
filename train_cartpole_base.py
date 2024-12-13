import json
import sys
from datetime import datetime, timezone

import numpy as np
import torch
from torchrl.envs.utils import (
    ExplorationType,
    set_exploration_type,
)
from torchrl.envs.libs.dm_control import DMControlEnv
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.transforms import (
    TransformedEnv,
    CatTensors,
    DoubleToFloat,
    Compose,
    Reward2GoTransform,
    RenameTransform,
    StepCounter,
)
from transforms import TrajCounter
from torchrl.collectors import SyncDataCollector
from tqdm import tqdm
from agents import CartpoleAgent

import wandb
import yaml


def train_base(config, interactive=None):

    def constraint_transform(td):
        # Constraint reward:
        td["constraint"] = (td["velocity"] ** 2).sum(-1, keepdim=True)
        return td

    env = TransformedEnv(
        DMControlEnv("cartpole", "swingup"),
        Compose(
            CatTensors(
                in_keys=["position", "velocity"], out_key="state", del_keys=False
            ),
            DoubleToFloat(in_keys=["state"]),
            RenameTransform(
                in_keys=["reward"], out_keys=["normal_reward"], create_copy=True
            ),
            StepCounter(),
            # TrajCounter(out_key="traj_count"),  # To count which episode we are on
            constraint_transform,
        ),
    )
    # env.append_transform(TrajCounter())
    # r = env.rollout(18, break_when_any_done=False)
    # r["next", "traj_count"]

    agent = CartpoleAgent(
        n_states=5,
        n_actions=1,
        action_spec=env.action_spec,
        num_optim_epochs=config["num_optim_epochs"],
        buffer_size=config["buffer_size"],
        sub_batch_size=config["sub_batch_size"],
        device=config["device"],
        max_grad_norm=config["max_grad_norm"],
        policy_lr=config["policy_lr"],
        qvalue_lr=config["qvalue_lr"],
        gamma=config["gamma"],
        target_eps=config["target_eps"],
        mode="train",
    )
    t = Compose(
        Reward2GoTransform(
            gamma=config["gamma"],
            in_keys=[("next", "reward")],
            out_keys=["reward_to_go"],
        ),
        Reward2GoTransform(
            gamma=config["gamma"],
            in_keys=[("constraint")],
            out_keys=["constraint_to_go"],
        ),
    )
    collector = SyncDataCollector(
        env,
        agent.policy,
        frames_per_batch=config["batch_size"],
        total_frames=config["total_frames"],
        split_trajs=False,
        device=config["device"],
        postproc=t.inv,
    )

    pbar = tqdm(total=collector.total_frames)

    for i, td in enumerate(collector):
        losses, max_grad_norm = agent.process_batch(td)

        wandb.log(
            {
                "loss_actor": losses["loss_actor"].item(),
                "loss_qvalue": losses["loss_qvalue"].item(),
                "loss_alpha": losses["loss_alpha"].item(),
                "mean reward": np.mean(td["next", "reward"].cpu().numpy()),
                "max_grad_norm": max_grad_norm,
                "mean reward_to_go": np.mean(td["reward_to_go"].cpu().numpy()),
                "mean constraint_to_go": np.mean(td["constraint_to_go"].cpu().numpy()),
                "Q-value network norm": torch.norm(
                    torch.stack(
                        [
                            torch.norm(p)
                            for p in agent.loss_module.qvalue_network_params.parameters()
                        ]
                    )
                ).item(),
                "buffer size": len(agent.replay_buffer),
                "Batch number": i,
            }
        )
        pbar.update(td.numel())


if __name__ == "__main__":
    # Treat first argument of program as path to YAML file with config
    with open(sys.argv[1], encoding="UTF-8") as f:
        config = yaml.safe_load(f)

    wandb.init(
        project="cartpole-base-train",
        name=f"cartpole-base-train|{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')}",
        config=config,
    )
    train_base(config)
