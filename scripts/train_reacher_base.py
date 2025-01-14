import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

import torch

import wandb
from agents.base_agents import ReacherSACAgent, ReacherDDPGAgent
from trainers import OffpolicyTrainer
from envs.dm_env import get_reacher_env

wandb.init(project="reacher-base")

device = torch.device("cpu")
batch_size = 64
gamma = 0.99
max_steps = 1000

env = get_reacher_env(device=device, constraint_weight=1.0, max_steps=max_steps)

agent = ReacherDDPGAgent(
    device=device,
    batch_size=batch_size,
    sub_batch_size=batch_size,
    num_epochs=1,
    replay_buffer_args={
        "buffer_size": 1_000_000,
        "min_buffer_size": 1_000,
        "alpha": 0.7,
        "beta": 0.5
    },
    max_grad_norm=1.0,
    env=env,
    agent_detail_args={
        "agent_gamma": gamma,
        "target_eps": 0.999,
        "lr": 1e-4
    }
)


trainer = OffpolicyTrainer(
    env=env,
    agent=agent,
    progress_bar=True,
    times_to_eval=20,
    collector_device=device,
    log=True,
    max_eval_steps=max_steps,
    collector_args={
        "batch_size": batch_size,
        "total_frames": 1_000_000,
    },
    env_gamma=gamma
)
trainer.train(when_constraints_active=0.99)
