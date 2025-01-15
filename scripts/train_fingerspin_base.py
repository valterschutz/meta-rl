import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

import torch

import wandb
from agents.base_agents import ReacherSACAgent, ReacherDDPGAgent, CartpoleDDPGAgent, CartpoleTD3Agent, FingerspinTD3Agent
from trainers import OffpolicyTrainer
from envs.dm_env import get_reacher_env, get_cartpole_env, get_fingerspin_env
from torch import nn

os.environ['MUJOCO_GL'] = 'egl'

wandb.init(project="fingerspin-base")

device = torch.device("cpu")
batch_size = 64
gamma = 0.99
max_steps = 500

env = get_fingerspin_env(constraint_weight=1.0, max_steps=max_steps, device=device)
pixel_env = get_fingerspin_env(constraint_weight=1.0, max_steps=max_steps, device=device, from_pixels=True, pixels_only=False)

agent = FingerspinTD3Agent(
    device=device,
    batch_size=batch_size,
    sub_batch_size=batch_size,
    num_epochs=1,
    replay_buffer_args={
        "buffer_size": 1_000_000,
        "min_buffer_size": 10_000,
        "alpha": 0.7,
        "beta": 0.5
    },
    env=env,
    agent_detail_args={
        "agent_gamma": gamma,
        "target_eps": 0.999,
        "actor_lr": 1e-4,
        "value_lr": 1e-4,
        "actor_max_grad": 10,
        "value_max_grad": 10,
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
    env_gamma=gamma,
    eval_env=pixel_env
)
trainer.train(when_constraints_active=0.99)
