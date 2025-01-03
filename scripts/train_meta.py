import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

import torch

import wandb
from trainers import train_meta_agent

wandb.init(project="clean-meta-sac")
returns = train_base_agent(
        device=torch.device("cpu"),
        n_base_episodes=100,
        log=True,
        progress_bar=True,
        batch_size = 200,
        sub_batch_size = 20,
        num_epochs = 100
)
