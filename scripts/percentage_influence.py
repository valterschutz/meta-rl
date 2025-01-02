import os
import sys

from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

import pandas as pd
import torch

import wandb
from agents import train_base_agent

n_evals = 5
percentages = [i/10 for i in range(10)]
# n_evals = 2
# percentages = [0.2, 0.5]
rows = [] # A list of tuples where each tuple contains (percentage, n_eval, return)

try:
    pbar = tqdm(total=len(percentages)*n_evals)
    for percentage in percentages:
        for i in range(n_evals):
            returns = train_base_agent(
                    device=torch.device("cpu"),
                    total_frames=10_000,
                    min_buffer_size=200,
                    n_states=10,
                    return_x=5,
                    return_y=1,
                    percentage_constraints_active=percentage,
                    times_to_eval=10,
                    log=False,
                    progress_bar=False
            )
            rows.append((percentage, i, sum(returns)/len(returns)))
            pbar.update(1)
except KeyboardInterrupt:
    pbar.close()
    print("Interrupted")

df = pd.DataFrame(rows, columns=["percentage", "n_eval", "mean_return"])
# Save the dataframe to a csv file, relative to this file's location
df.to_csv(os.path.join(os.path.dirname(__file__), "../data/percentage_influence.csv"))
