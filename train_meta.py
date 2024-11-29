# TODO:
# - [X] Make sure that base agent loss converges in each meta episode before applying meta actions

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
from utils import DictWrapper


def log(pbar, meta_td, episode, step):
    # TODO
    pbar.update(meta_td.numel())
    wandb.log(
        {
            f"episode-{episode}/step": step,
            f"episode-{episode}/meta state 1": meta_td["state"][0].item(),
            f"episode-{episode}/meta state 2": meta_td["state"][1].item(),
            f"episode-{episode}/meta action": meta_td["action"].item(),
        }
    )


with open(sys.argv[1], "r") as f:
    base_config = json.load(f)
with open(sys.argv[2], "r") as f:
    meta_config = json.load(f)

base_env, base_agent, base_collector = get_base_from_config(DictWrapper(base_config))


# Meta env
meta_env = MetaEnv(
    base_env=base_env,
    base_agent=base_agent,
    base_collector=base_collector,
    device=meta_config["device"],
)
check_env_specs(meta_env)

# Meta agent
meta_agent = MetaAgent(
    state_spec=meta_env.state_spec,
    action_spec=meta_env.action_spec,
    device=meta_config["device"],
    max_grad_norm=meta_config["max_grad_norm"],
    lr=meta_config["lr"],
    hidden_units=meta_config["hidden_units"],
    clip_epsilon=meta_config["clip_epsilon"],
    use_entropy=meta_config["use_entropy"],
    gamma=meta_config["gamma"],
    lmbda=meta_config["lmbda"],
)

# Try to do a rollout
meta_td = meta_env.rollout(1000, meta_agent.policy)
print(f"meta_td: {meta_td}")
meta_agent.process_batch(meta_td)
meta_td = meta_env.rollout(1000, meta_agent.policy)
# fail

wandb.login()
wandb.init(
    project="toy-meta-train",
    name=f"toy-meta-train|{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    config={
        **{f"meta-{k}": v for k, v in meta_config.items()},
        **{f"base-{k}": v for k, v in base_config.items()},
    },
)

meta_steps_per_episode = base_config["total_frames"] // base_config["batch_size"]
meta_total_steps = meta_steps_per_episode * meta_config["episodes"]
pbar = tqdm(total=meta_total_steps)

meta_collector = SyncDataCollector(
    meta_env,
    meta_agent.policy,
    frames_per_batch=1,
    total_frames=meta_total_steps,
    device=meta_config["device"],
)

for meta_td in meta_collector:
    meta_loss = meta_agent.process_batch(meta_td)
    meta_td = meta_td.squeeze(0)
    # log(pbar, meta_td, episode=0, step=0)
    wandb.log(
        {
            # "meta loss": meta_agent.loss,
            "meta reward": meta_td["next", "reward"].sum().item(),
            "meta state 1": meta_td["state"][0].item(),
            "meta state 2": meta_td["state"][1].item(),
            "meta action": meta_td["action"].item(),
        }
    )
    pbar.update(meta_td.numel())

# for i in range(meta_config["episodes"]):
#     meta_td = meta_env.reset()  # Resets base agent in meta environment
#     for j in range(meta_steps):
#         # print(f"td before applying meta action: {meta_td}")
#         meta_td = meta_agent.policy(meta_td)
#         # print(f"td after applying meta action: {meta_td}")
#         # fail
#         meta_td = meta_env.step(meta_td)
#         meta_agent.process_batch(meta_td.unsqueeze(0))
#         # with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
#         #     assert not base_env.constraints_enabled  # TODO: remove
#         #     base_eval_td = base_env.rollout(100, base_agent.policy)
#         log(pbar, meta_td.detach().cpu(), episode=i, step=j)
#         meta_td = step_mdp(meta_td)
