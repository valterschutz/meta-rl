import numpy as np
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F

import os
import sys
import pickle

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from envs.toy_env import ToyEnv

def transposed_dict_iter(d: dict):
    """
    Given data in the format
    ```
    data = {
        'a': [1, 2, 3],
        'b': [4, 5, 6],
        'c': [7, 8, 9]
    }
    ```
    return an iterator that yields
    ```
    {'a': 1, 'b': 4, 'c': 7}
    {'a': 2, 'b': 5, 'c': 8}
    {'a': 3, 'b': 6, 'c': 9}
    ```
    """

    # Transpose the lists (retrieve elements by index across all keys)
    transposed = zip(*d.values())

    # Iterate through the transposed structure
    for elements in transposed:
        result = {key: value for key, value in zip(d.keys(), elements)}
        yield result


date = "2025-01-31"
n_states = 16

# Load pickles
with open(f"data/{date}|{n_states}-states.pkl", "rb") as f:
    pkl = pd.read_pickle(f)

# Create an identical environment as during training
env = ToyEnv.get_env(pkl["env_config"])

# Policy matrices for "slow" and "fast" policies, excluding the terminal state
optimal_non_constrained_policy = env.calc_optimal_policy(constraints_active=False)
optimal_constrained_policy = env.calc_optimal_policy(constraints_active=True)

d = []
for type, multiple_run_data in pkl["multiple_run_data"].items():
    for run, history in enumerate(multiple_run_data):
        for batch, metrics in enumerate(transposed_dict_iter(history)):
            d.append({
                "run": run,
                "batch": batch,
                "type": type,
                **metrics
            })


# Add a new column "distance to optimal policy" which is "distance to slow policy" if "type" is "constrained" and "distance to fast policy" if "type" is "unconstrained"
for i in range(len(d)):
    if d[i]["type"] == "constrained":
        d[i]["optimal qvalue offset"] = float(d[i]["constrained qvalue offset"])
        d[i]["optimal policy offset"] = float(d[i]["constrained policy offset"])
    else:
        d[i]["optimal qvalue offset"] = float(d[i]["non-constrained qvalue offset"])
        d[i]["optimal policy offset"] = float(d[i]["non-constrained policy offset"])



# Convert into dataframe
df = pd.DataFrame(d)

# Normalize the "optimal policy offsets" such that they all start at 1 at batch 0
for (run, type_), group in df.groupby(["run", "type"]):
    initial_value = group[group["batch"] == 0]["optimal policy offset"].values[0]
    df.loc[(df["run"] == run) & (df["type"] == type_), "optimal policy offset"] /= initial_value

# sns.lineplot(
#     data=df,
#     x="batch",
#     y="non-constrained qvalue offset",
#     hue="type",
# )
# plt.show()

# sns.lineplot(
#     data=df,
#     x="batch",
#     y="constrained qvalue offset",
#     hue="type",
# )

# sns.lineplot(
#     data=df,
#     x="batch",
#     y="non-constrained policy offset",
#     hue="type",
# )

# sns.lineplot(
#     data=df,
#     x="batch",
#     y="constrained policy offset",
#     hue="type",
# )

sns.lineplot(
    data=df,
    x="batch",
    y="optimal policy offset",
    hue="type",
)
plt.show()
