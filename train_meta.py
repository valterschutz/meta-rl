import json
import sys
from datetime import datetime

import torch
from torchrl.envs.utils import (
    ExplorationType,
    check_env_specs,
    set_exploration_type,
    step_mdp,
)
from torchrl.collectors import SyncDataCollector
from tqdm import tqdm

import wandb
from agents import MetaAgent
from base import get_base_from_config
from env import MetaEnv
from utils import DictWrapper, MethodLogger
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("base_config", type=str)
parser.add_argument("meta_config", type=str)
args = parser.parse_args()
with open(args.base_config, "r", encoding="UTF-8") as f:
    base_config = json.load(f)
with open(args.meta_config, "r", encoding="UTF-8") as f:
    meta_config = json.load(f)

base_env, base_agent, base_collector_fn = get_base_from_config(DictWrapper(base_config))

# Meta env
meta_env = MetaEnv(
    base_env=base_env,
    base_agent=base_agent,
    base_collector_fn=base_collector_fn,
    device=meta_config["device"],
)
check_env_specs(meta_env)

# Meta agent
meta_agent = MetaAgent(
    state_spec=meta_env.state_spec,
    action_spec=meta_env.action_spec,
    num_optim_epochs=1,
    buffer_size=1,
    sub_batch_size=1,
    device=meta_config["device"],
    max_grad_norm=meta_config["max_grad_norm"],
    lr=meta_config["lr"],
    gamma=meta_config["gamma"],
    lmbda=meta_config["lmbda"],
    clip_epsilon=meta_config["clip_epsilon"],
    use_entropy=meta_config["use_entropy"],
    hidden_units=meta_config["hidden_units"],
)

meta_steps_per_episode = base_config["total_frames"] // base_config["batch_size"]
meta_total_steps = meta_steps_per_episode * meta_config["episodes"]
pbar = tqdm(total=meta_total_steps)

wandb.login()
wandb.init(
    project="toy-meta-train",
    name=f"toy-meta-train|{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    config={
        **{f"meta-{k}": v for k, v in meta_config.items()},
        **{f"base-{k}": v for k, v in base_config.items()},
    },
)

for i in range(meta_config["episodes"]):
    td = meta_env.reset()  # Resets base agent in meta environment
    for j in range(meta_steps_per_episode):
        td = meta_agent.policy(td)
        td = meta_env.step(td)
        losses, max_grad = meta_agent.process_batch(td.unsqueeze(0))
        pbar.update(td.numel())
        wandb.log(
            {
                "step": j,
                "state 1": td["state"][0].item(),
                "state 2": td["state"][1].item(),
                "action": td["action"].item(),
                "reward": td["next", "reward"].item(),
                "loss_objective": losses["loss_objective"].item(),
                "loss_critic": losses["loss_critic"].item(),
                "loss_entropy": losses["loss_entropy"].item(),
                "base_agent loss_objective": td[
                    "base_agent_losses", "loss_objective"
                ].item(),
                "base_agent loss_critic": td["base_agent_losses", "loss_critic"].item(),
                "base_agent loss_entropy": td[
                    "base_agent_losses", "loss_entropy"
                ].item(),
            }
        )
        td = step_mdp(td)
