# TODO:
# - [X] Make sure that base agent loss converges in each meta episode before applying meta actions

from tqdm import tqdm
from datetime import datetime

from torchrl.envs.utils import (
    check_env_specs,
    step_mdp,
    set_exploration_type,
    ExplorationType,
)
from torchrl.collectors import SyncDataCollector

import torch
import wandb

from env import get_base_env, MetaEnv
from agents import BaseAgent, MetaAgent
from utils import log

return_x = 0.2  # Optimal return using slow path
return_y = 0.1  # Return for using fast path
big_reward = 10.0
n_states = 20
gamma = 0.9
rollout_timeout = 10 * n_states
meta_episodes = 2
meta_steps_per_episode = 100  # TODO: 10 should be enough
device = torch.device("cpu")
gamma = 0.9

# Base env
base_env = get_base_env(
    left_reward=0,
    right_reward=0,
    down_reward=0,
    up_reward=0,
    n_pos=n_pos,
    big_reward=big_reward,
    random_start=False,
    punishment=0.0,
    seed=None,
    device="cpu",
    constraints_enabled=True,
).to(device)
check_env_specs(base_env)

# Base agent
base_agent = BaseAgent(
    state_spec=base_env.state_spec,
    action_spec=base_env.action_spec,
    num_optim_epochs=10,
    buffer_size=20,
    sub_batch_size=20,
    device="cpu",
    max_grad_norm=1,
    lr=1e-2,
    gamma=gamma,
    lmbda=0.5,
)

# Meta env
meta_env = MetaEnv(
    base_env=base_env, base_agent=base_agent, total_frames=1_000, device=device
)
check_env_specs(meta_env)

# Meta agent
meta_agent = MetaAgent(
    state_spec=meta_env.state_spec,
    action_spec=meta_env.action_spec,
    device=device,
    max_grad_norm=1,
    lr=1e-2,
)

wandb.login()
wandb.init(
    project="meta_toy",
    name=f"meta_toy|{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    config={
        "base_agent.buffer_size": base_agent.buffer_size,
        "n_states": n_states,
    },
)

pbar = tqdm(total=meta_steps_per_episode * meta_episodes)

for i in range(meta_episodes):
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
