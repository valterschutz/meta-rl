import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

import torch

import wandb
from trainers import train_toy_base_agent

wandb.init(project="clean-base-sac")
n_states = 102
gamma = 0.999
return_x = 9
return_y = 1
returns = train_toy_base_agent(
    device=torch.device("cpu"),
    total_frames=20_000,
    min_buffer_size=0,
    n_states=n_states,
    big_reward = 10,
    gamma = gamma,
    shortcut_steps=4,
    return_x=return_x,
    return_y=return_y,
    when_constraints_active=0.01,
    times_to_eval=20,
    log=True,
    progress_bar=True,
    batch_size = 50,
    sub_batch_size = 20,
    num_epochs = 10,
    agent_alg="SAC"
)
