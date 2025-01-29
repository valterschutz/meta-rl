import numpy as np
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F

date = "2025-01-29"
n_states = 10

# Load two pickles
with open(f"data/unconstrained_result_dicts|{date}|{n_states}-states.pkl", "rb") as f:
    unconstrained_result_dicts = pd.read_pickle(f)
with open(f"data/constrained_result_dicts|{date}|{n_states}-states.pkl", "rb") as f:
    constrained_result_dicts = pd.read_pickle(f)
# The pickles are lists ("run" dimension) of dictionaries.

# Policy matrices for "slow" and "fast" policies, excluding the terminal state
slow_policy_matrix = torch.zeros(n_states-1, 4)
slow_policy_matrix[:,1] = 1
fast_policy_matrix = torch.zeros(n_states-1, 4)
fast_policy_matrix[:,3] = 1

def qvalues_to_policy_matrix(qvalues):
    indices = qvalues.argmax(dim=-1)
    policy = F.one_hot(indices, num_classes=4)
    # Exclude the terminal state
    return policy[:-1]

distances = []
for run, result_dict in enumerate(constrained_result_dicts):
    for batch, qvalues in enumerate(result_dict["qvalues"]):
        # qvalues = result_dict["qvalues"]
        policy_matrix = qvalues_to_policy_matrix(qvalues)
        distances.append({
            "batch": batch,
            "run": run,
            "distance to slow policy": torch.norm(policy_matrix - slow_policy_matrix).item(),
            "distance to fast policy": torch.norm(policy_matrix - fast_policy_matrix).item(),
            "type": "constrained",
        })

for run, result_dict in enumerate(unconstrained_result_dicts):
    for batch, qvalues in enumerate(result_dict["qvalues"]):
        # qvalues = result_dict["qvalues"]
        policy_matrix = qvalues_to_policy_matrix(qvalues)
        distances.append({
            "batch": batch,
            "run": run,
            "distance to slow policy": torch.norm(policy_matrix - slow_policy_matrix).item(),
            "distance to fast policy": torch.norm(policy_matrix - fast_policy_matrix).item(),
            "type": "unconstrained",
        })

# Add a new column "distance to optimal policy" which is "distance to slow policy" if "type" is "constrained" and "distance to fast policy" if "type" is "unconstrained"
for i in range(len(distances)):
    if distances[i]["type"] == "constrained":
        distances[i]["distance to optimal policy"] = distances[i]["distance to slow policy"]
    else:
        distances[i]["distance to optimal policy"] = distances[i]["distance to fast policy"]

# Convert into dataframe
df = pd.DataFrame(distances)

# fig, axes = plt.subplots(1, 2, figsize=(12, 6))


sns.lineplot(
    data=df,
    x="batch",
    y="distance to slow policy",
    hue="type",
)
plt.show()

sns.lineplot(
    data=df,
    x="batch",
    y="distance to fast policy",
    hue="type",
)
plt.show()
