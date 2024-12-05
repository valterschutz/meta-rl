import json
import pickle
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
import tensordict
import yaml


parser = argparse.ArgumentParser()
parser.add_argument("base_config", type=str)
parser.add_argument("meta_config", type=str)
args = parser.parse_args()
with open(args.base_config, "r", encoding="UTF-8") as f:
    base_config = yaml.safe_load(f)
with open(args.meta_config, "r", encoding="UTF-8") as f:
    meta_config = yaml.safe_load(f)

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
meta_total_steps = meta_steps_per_episode * meta_config["train_episodes"]
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

for i in range(meta_config["train_episodes"]):
    meta_td = meta_env.reset()  # Resets base agent in meta environment
    for j in range(meta_steps_per_episode):
        meta_td = meta_agent.policy(meta_td)
        meta_td = meta_env.step(meta_td)
        meta_losses, meta_max_grad = meta_agent.process_batch(meta_td.unsqueeze(0))
        pbar.update(meta_td.numel())
        wandb.log(
            {
                "step": j,
                "base_mean_reward": meta_td["base_mean_reward"].item(),
                "base_std_reward": meta_td["base_std_reward"].item(),
                "last_action": meta_td["last_action"].item(),
                "action": meta_td["action"].item(),
                "meta reward": meta_td["next", "reward"].item(),
                "meta loss_objective": meta_losses["loss_objective"].item(),
                "meta loss_critic": meta_losses["loss_critic"].item(),
                "meta loss_entropy": meta_losses["loss_entropy"].item(),
                "meta max_grad_norm": meta_max_grad,
                "base loss_objective": meta_td[
                    "base", "losses", "loss_objective"
                ].item(),
                "base loss_critic": meta_td["base", "losses", "loss_critic"].item(),
                "base loss_entropy": meta_td["base", "losses", "loss_entropy"].item(),
                "base state distribution": wandb.Histogram(meta_td["base", "states"]),
                "base reward distribution": wandb.Histogram(meta_td["base", "rewards"]),
                "base true_reward distribution": wandb.Histogram(
                    meta_td["base", "true_rewards"]
                ),
            }
        )
        meta_td = step_mdp(meta_td)

# Save meta agent
print(f"Saving meta agent to models/{meta_config['policy_module_name']}.pth")
torch.save(
    meta_agent.policy_module.state_dict(),
    f"models/{meta_config['policy_module_name']}.pth",
)
torch.save(
    meta_agent.value_module.state_dict(),
    f"models/{meta_config['value_module_name']}.pth",
)
