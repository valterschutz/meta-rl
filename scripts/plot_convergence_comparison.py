import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load two pickles
with open("data/unconstrained_result_dicts|2025-01-27|10-states.pkl", "rb") as f:
    unconstrained_result_dicts = pd.read_pickle(f)
with open("data/constrained_result_dicts|2025-01-27|10-states.pkl", "rb") as f:
    constrained_result_dicts = pd.read_pickle(f)
# The pickles are lists ("run" dimension) of dictionaries. Each dictionary contains a key "train_info_dicts" which is itself a list ("batch" dimension) of dictionaries, each of which contains a key "distance to slow policy" which is a float. We want to extract the "distance to slow policy" values and store them in a dataframe. The final dataframe should have the columns "batch", "run", "distance to slow policy", and "type" (which is either "unconstrained" or "constrained").

distances = []
for run, result_dict in enumerate(unconstrained_result_dicts):
    for batch, train_info_dict in enumerate(result_dict["train_info_dicts"]):
        distances.append({
            "batch": batch,
            "run": run,
            "distance to slow policy": train_info_dict["distance to slow policy"],
            "distance to fast policy": train_info_dict["distance to fast policy"],
            "type": "unconstrained",
        })
for run, result_dict in enumerate(constrained_result_dicts):
    for batch, train_info_dict in enumerate(result_dict["train_info_dicts"]):
        distances.append({
            "batch": batch,
            "run": run,
            "distance to slow policy": train_info_dict["distance to slow policy"],
            "distance to fast policy": train_info_dict["distance to fast policy"],
            "type": "constrained",
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
    y="distance to optimal policy",
    hue="type",
    # ax=axes[0]
)
# sns.lineplot(
#     data=df,
#     x="batch",
#     y="distance to fast policy",
#     hue="type",
#     ax=axes[1]
# )
plt.show()
