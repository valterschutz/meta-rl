import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

import torch

import wandb
from agents import train_base_agent

wandb.init(project="clean-base-sac")
returns = train_base_agent(
        device=torch.device("cpu"),
        total_frames=50_000,
        min_buffer_size=1_000,
        n_states=10,
        percentage_constraints_active=0.5,
        times_to_eval=10,
        log=True,
        progress_bar=True
)
print(f"returns: {returns}")
