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


n_meta_episodes = 10
device = torch.device("cpu")
n_actions = 4
lr = 1e-1
times_to_eval = 10
# eval_every_n_epoch = (total_frames // batch_size) // times_to_eval
# max_rollout_steps = n_pos * 3
optimal_return = 0.1
gap = 0.1
big_reward = 10.0
n_pos = 20

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

# Base agent
base_agent = BaseAgent(
    state_spec=base_env.state_spec,
    action_spec=base_env.action_spec,
    num_optim_epochs=10,
    buffer_size=100,
    sub_batch_size=20,
    device="cpu",
    max_grad_norm=1,
    lr=1e-1,
)

# Meta env
meta_env = MetaEnv(
    base_env=base_env, base_agent=base_agent, total_frames=1000, device="cpu"
)

# Meta agent
meta_agent = MetaAgent(
    state_spec=meta_env.state_spec,
    action_spec=meta_env.action_spec,
    device="cpu",
    max_grad_norm=1,
    lr=1e-1,
)

meta_collector = SyncDataCollector(
    meta_env,
    meta_agent.policy,
    frames_per_batch=1,
    total_frames=100,
    split_trajs=False,
    device="cpu",
)

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
        "meta_collector.total_frames": meta_collector.total_frames,
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

pbar = tqdm(total=meta_collector.total_frames)

for i, meta_td in enumerate(meta_collector):
    meta_agent.process_batch(meta_td)
    # Update logs
    log(pbar, meta_td)


# After training, do a single rollout and print all states and actions
# with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
#     td = env.rollout(max_rollout_steps, base_policy)
#     for i in range(len(td)):
#         print(
#             f"Step {i}: pos={td['pos'][i].item()}, action={td['action'][i].item()}, reward={td['next','reward'][i].item()} done={td['next','done'][i].item()}"
#         )
