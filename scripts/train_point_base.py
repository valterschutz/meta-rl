import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

import torch

import wandb
from trainers import train_point_base_agent

wandb.init(project="point-base")
returns = train_point_base_agent(
    device=torch.device("cpu"),
    total_frames=5_000,
    min_buffer_size=0,
    when_constraints_active=0.99,
    times_to_eval=20,
    log=True,
    progress_bar=True,
    batch_size = 200,
    sub_batch_size = 20,
    num_epochs = 100
)
