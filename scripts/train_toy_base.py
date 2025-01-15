import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

import torch

import wandb
from trainers import OffpolicyTrainer
from agents.base_agents import ToyDQNAgent
from envs.toy_env import ToyEnv

from torchrl.envs.transforms import Compose, StepCounter, DoubleToFloat, TransformedEnv, DTypeCastTransform

os.environ['MUJOCO_GL'] = 'egl'

wandb.init(project="toy-base")

device = torch.device("cpu")
batch_size = 64
gamma = 0.999
n_states = 500
max_steps = 5*n_states
shortcut_steps = 2
big_reward = 10
return_x = 5
return_y = 1
total_frames = 100_000

x, y = ToyEnv.calculate_xy(n_states=n_states, shortcut_steps=shortcut_steps, return_x=return_x, return_y=return_y, big_reward=big_reward, punishment=0, gamma=gamma)

env = ToyEnv(
    left_reward=x,
    right_reward=x,
    down_reward=y,
    up_reward=y,
    n_states=n_states,
    shortcut_steps=shortcut_steps,
    big_reward=big_reward,
    punishment=0.0,
    constraints_active=False,
    random_start=False,
    seed=None,
    device=device)
env = TransformedEnv(
    env,
    Compose(
        StepCounter(max_steps=max_steps),
        DTypeCastTransform(dtype_in=torch.long, dtype_out=torch.float32, in_keys=["observation"]),
    )
)

agent = ToyDQNAgent(
    device=device,
    batch_size=batch_size,
    sub_batch_size=batch_size,
    num_epochs=1,
    replay_buffer_args={
        "buffer_size": total_frames,
        "min_buffer_size": 0,
        "alpha": 0.7,
        "beta": 0.5
    },
    env=env,
    agent_detail_args={
        "agent_gamma": gamma,
        "target_eps": 0.99,
        "value_lr": 1e-3,
        "value_max_grad": 10,
        "num_cells": [32, 32],
        "qvalue_eps": 0.1
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
        "total_frames": total_frames,
    },
    env_gamma=gamma,
    eval_env=None
)
trainer.train(when_constraints_active=0.99)
