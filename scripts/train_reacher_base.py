import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

import torch

import wandb
from trainers import train_reacher_base_agent

wandb.init(project="reacher-base")
returns = train_reacher_base_agent(
    device=torch.device("cuda"),
    total_frames=1_500_000,
    min_buffer_size=0,
    when_constraints_active=0.99,
    times_to_eval=10,
    log=True,
    progress_bar=True,
    batch_size = 1_000,
    sub_batch_size = 100,
    num_epochs = 100
)
