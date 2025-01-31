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
    # unconstrained_result_dicts pkl["unconstrained_result_dicts"]
    # constrained_result_dicts = pkl["constrained_result_dicts"]
    # toggled_result_dicts = pkl["toggled_result_dicts"]
    # env_config = pkl["env_config"]

# Create an identical environment as during training
env = ToyEnv.get_env(pkl["env_config"])

# The pickles are lists ("run" dimension) of dictionaries.

# Policy matrices for "slow" and "fast" policies, excluding the terminal state
optimal_non_constrained_policy = env.calc_optimal_policy(constraints_active=False)
optimal_constrained_policy = env.calc_optimal_policy(constraints_active=True)


# def qvalues_to_policy_matrix(qvalues):
#     indices = qvalues.argmax(dim=-1)
#     policy = F.one_hot(indices, num_classes=4)
#     # Exclude the terminal state
#     return policy[:-1]

# The pickle was saved using
#

# pickle.dump({
#     "multiple_run_data": {
#         "unconstrained": unconstrained_multiple_run_data,
#         "constrained": constrained_multiple_run_data,
#         "toggled": toggled_multiple_run_data
#     },
#     "env_config": env_config,
# }, f)
# where the "multiple_run_data" are lists of histories, each element pertaining to the history of a specific run. The histories are dictionaries and contain various string keys, for example "slow_dst" and "fast_dst", and the values are lists of metrics during the run.

# From this data, create a pandas dataframe with the columns "run", "batch", "distance to slow policy", "distance to fast policy", "type" (either "constrained" or "unconstrained")

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
# for i in range(len(d)):
#     if d[i]["type"] == "constrained":
#         d[i]["optimal qvalue offset"] = d[i]["constrainted qvalue offset"]
#     else:
#         d[i]["optimal qvalue offset"] = d[i]["constrainted qvalue offset"]

# Convert into dataframe
df = pd.DataFrame(d)


sns.lineplot(
    data=df,
    x="batch",
    y="constrained qvalue offset",
    hue="type",
)
plt.show()

sns.lineplot(
    data=df,
    x="batch",
    y="non-constrained qvalue offset",
    hue="type",
)
plt.show()
