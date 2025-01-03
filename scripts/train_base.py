import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

import torch

import wandb
from trainers import train_base_agent

wandb.init(project="clean-base-sac")
n_states = 102
return_x = 10*0.99**50 - 1e-3
returns = train_base_agent(
    device=torch.device("cpu"),
    total_frames=50_000,
    min_buffer_size=0,
    n_states=n_states,
    shortcut_steps=5,
    return_x=return_x,
    return_y=-100,
    when_constraints_active=0.99,
    times_to_eval=20,
    log=True,
    progress_bar=True,
    batch_size = 200,
    sub_batch_size = 20,
    num_epochs = 100
)
