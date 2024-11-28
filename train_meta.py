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
from tqdm import tqdm

import wandb
from agents import MetaAgent
from base import get_base_from_config
from env import MetaEnv
from utils import log, DictWrapper

device = "cpu"

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
    device=device,
)
check_env_specs(meta_env)

# Meta agent
meta_agent = MetaAgent(
    state_spec=meta_env.state_spec,
    action_spec=meta_env.action_spec,
    device=device,
    max_grad_norm=meta_config["max_grad_norm"],
    lr=meta_config["lr"],
)

wandb.login()
wandb.init(
    project="toy-meta-train",
    name=f"toy-meta-train|{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    config={
        **{f"meta-{k}": v for k, v in meta_config.items()},
        **{f"base-{k}": v for k, v in base_config.items()},
    },
)

pbar = tqdm(total=meta_config["steps_per_episode"] * meta_config["episodes"])

for i in range(meta_config.episodes):
    meta_td = meta_env.reset()  # Resets base agent in meta environment
    for j in range(meta_steps_per_episode):
        meta_td = meta_agent.policy(meta_td)
        meta_td = meta_env.step(meta_td)
        meta_agent.process_batch(meta_td.unsqueeze(0))
        # Update logs, including a base agent evaluation
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            assert not base_env.constraints_enabled  # TODO: remove
            base_eval_td = base_env.rollout(100, base_agent.policy)
        log(pbar, meta_td.detach().cpu(), base_eval_td.cpu(), episode=i, step=j)
        meta_td = step_mdp(meta_td)
