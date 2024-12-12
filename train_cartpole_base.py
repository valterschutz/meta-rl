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
)
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
            constraint_transform,
        ),
    )

    agent = CartpoleAgent(
        n_states=5,
        n_actions=1,
        action_spec=env.action_spec,
        num_optim_epochs=config["num_optim_epochs"],
        buffer_size=config["buffer_size"],
        sub_batch_size=config["sub_batch_size"],
        device=config["device"],
        max_grad_norm=config["max_grad_norm"],
        lr=config["lr"],
        gamma=config["gamma"],
        lmbda=config["lmbda"],
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

    # Initialize weight snapshots
    prev_policy_weights = {
        name: param.clone() for name, param in agent.policy_module.named_parameters()
    }
    prev_qvalue_weights = {
        name: param.clone() for name, param in agent.qvalue_module.named_parameters()
    }
    repr_sample1 = env.reset()
    repr_sample2 = repr_sample1.clone()
    repr_sample1["action"] = torch.tensor([0.42], device=config["device"])
    repr_sample2["action"] = torch.tensor([0.9], device=config["device"])
    repr_sample1 = agent.qvalue_module(repr_sample1)
    repr_sample2 = agent.qvalue_module(repr_sample2)
    loss = (
        repr_sample1["state_action_value"] - repr_sample2["state_action_value"]
    ) ** 2
    loss.backward()
    print("loss: ", loss)
    agent.optim.step()
    agent.optim.zero_grad()
    # repr_td = repr_sample.clone()
    # repr_td = agent.qvalue_module(repr_td)
    # Calculate difference in Q-value weights
    qvalue_ssd = sum(
        torch.sum((param - prev_qvalue_weights[name]) ** 2).item()
        for name, param in agent.qvalue_module.named_parameters()
    )
    print("Qvalue ssd: ", qvalue_ssd)

    for td in collector:
        losses, max_grad_norm = agent.process_batch(td)

        # print("Qmodule output: ", repr_td["state_action_value"])
        # agent.optim.step()
        # agent.optim.zero_grad()

        # Calculate weight changes
        policy_ssd = sum(
            torch.sum((param - prev_policy_weights[name]) ** 2).item()
            for name, param in agent.policy_module.named_parameters()
        )
        qvalue_ssd = sum(
            torch.sum((param - prev_qvalue_weights[name]) ** 2).item()
            for name, param in agent.qvalue_module.named_parameters()
        )

        # Update previous weights for the next iteration
        prev_policy_weights = {
            name: param.clone()
            for name, param in agent.policy_module.named_parameters()
        }
        prev_qvalue_weights = {
            name: param.clone()
            for name, param in agent.qvalue_module.named_parameters()
        }

        wandb.log(
            {
                "loss_actor": losses["loss_actor"].item(),
                "loss_qvalue": losses["loss_qvalue"].item(),
                "loss_alpha": losses["loss_alpha"].item(),
                "mean reward": np.mean(td["next", "reward"].cpu().numpy()),
                "max_grad_norm": max_grad_norm,
                "mean reward_to_go": np.mean(td["reward_to_go"].cpu().numpy()),
                "mean constraint_to_go": np.mean(td["constraint_to_go"].cpu().numpy()),
                "policy ssd": policy_ssd,
                "qvalue ssd": qvalue_ssd,
                "buffer size": len(agent.replay_buffer),
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
