import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

import torch

import wandb
from trainers import train_base_agent

wandb.init(project="clean-base-sac")
returns = train_base_agent(
        device=torch.device("cpu"),
        total_frames=10_000,
        min_buffer_size=0,
        n_states=20,
        shortcut_steps=2,
        return_x=2,
        return_y=1,
        percentage_constraints_active=0.99,
        times_to_eval=10,
        log=True,
        progress_bar=True
)
