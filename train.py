# TODO:
# - [ ] Make sure that base agent loss converges in each meta episode before applying meta actions

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

meta_episodes = 2
meta_steps_per_episode = 100  # TODO: 10 should be enough
device = torch.device("cpu")
n_actions = 4

optimal_return = 0.1
gap = 0.1
big_reward = 10.0
n_pos = 10  # TODO: should be 20?

# (n_pos-1)*x + big_reward = optimal_return
x = (optimal_return - big_reward) / (n_pos - 1)
# (n_pos-1)/2*y + big_reward = optimal_return - gap
y = (optimal_return - gap - big_reward) / ((n_pos - 1) / 2)

# Base env
base_env = get_base_env(
    left_reward=x,
    right_reward=x,
    down_reward=y,
    up_reward=y,
    n_pos=n_pos,
    big_reward=big_reward,
    random_start=False,
    punishment=0.0,
    seed=None,
    device="cpu",
    constraints_enabled=False,
).to(device)
check_env_specs(base_env)

# Base agent
base_agent = BaseAgent(
    state_spec=base_env.state_spec,
    action_spec=base_env.action_spec,
    num_optim_epochs=10,
    buffer_size=100,
    sub_batch_size=20,
    device="cpu",
    max_grad_norm=1,
    lr=1e-2,  # TODO: should be 1e-1?
)

# Meta env
meta_env = MetaEnv(
    base_env=base_env, base_agent=base_agent, total_frames=1000, device="cpu"
)
check_env_specs(meta_env)

# Meta agent
meta_agent = MetaAgent(
    state_spec=meta_env.state_spec,
    action_spec=meta_env.action_spec,
    device="cpu",
    max_grad_norm=1,
    lr=1e-1,
)

# td = meta_env.reset()
# print(f"td after reset: {td=}")
# td = meta_agent.policy(td)
# print(f"td after policy: {td=}")
# td = meta_env.step(td)
# print(f"td after step: {td=}")
# print(f"next reward: {td['next', 'reward'].item()}")
# print(f"next state: {td['next', 'state'][0].item()}, {td['next', 'state'][1].item()}")
# fail
# meta_agent.process_batch(td.unsqueeze(0))

# meta_collector = SyncDataCollector(
#     meta_env,
#     meta_agent.policy,
#     frames_per_batch=1,
#     total_frames=100,
#     split_trajs=False,
#     device="cpu",
# )

# Test the meta collector
# it = iter(meta_collector)
# td = next(it)
# print(f"{td=}")
# fail

# all_pos = torch.arange(n_pos)
# td_all_pos = TensorDict(
#     {"pos": all_pos, "step_count": torch.zeros_like(all_pos)},
#     batch_size=[n_pos],
# ).to(device)

wandb.login()
wandb.init(
    project="meta_toy",
    name=f"meta_toy|{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    config={
        "base_agent.buffer_size": base_agent.buffer_size,
        "base_agent.sub_batch_size": base_agent.sub_batch_size,
        # "meta_collector.total_frames": total_frames,
        # "base num_optim_epochs": base_agent.num_optim_epochs,
        # "gamma": gamma,
        # "n_actions": n_actions,
        # "base_agent.max_grad_norm": base_agent.max_grad_norm,
        # "base_agent.lr": base_agent.lr,
        # "meta_agent.max_grad_norm": meta_agent.max_grad_norm,
        # "meta_agent.lr": meta_agent.lr,
        "left_reward": base_env.left_reward,
        "right_reward": base_env.right_reward,
        "down_reward": base_env.down_reward,
        "up_reward": base_env.up_reward,
        "n_pos": base_env.n_pos,
        "big_reward": base_env.big_reward,
        "punishment": base_env.punishment,
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
