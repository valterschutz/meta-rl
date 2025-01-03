import os
import sys

from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

import pandas as pd
import torch

from trainers import train_base_agent

n_evals = 4
percentages = [i/10 for i in range(10)]
rows = [] # A list of tuples where each tuple contains (percentage, n_eval, return)

try:
    pbar = tqdm(total=len(percentages)*n_evals)
    for percentage in percentages:
        for i in range(n_evals):
            returns = train_base_agent(
                device=torch.device("cpu"),
                total_frames=40_000,
                min_buffer_size=0,
                n_states=30,
                shortcut_steps=2,
                return_x=5,
                return_y=1,
                when_constraints_active=0.75,
                times_to_eval=20,
                log=False,
                progress_bar=False,
                batch_size = 200,
                sub_batch_size = 20,
                num_epochs = 100
            )
            # print(f"Percentage: {percentage}, n_eval: {i}, returns: {returns}")
            # rows.append((percentage, i, sum(returns)/len(returns)))
            rows.append((percentage, i, returns))
            pbar.update(1)
except KeyboardInterrupt:
    pbar.close()
    print("Interrupted")

df = pd.DataFrame(rows, columns=["percentage", "n_eval", "returns"])
# Save the dataframe to a csv file, relative to this file's location
df.to_csv(os.path.join(os.path.dirname(__file__), "../data/percentage_influence.csv"))
