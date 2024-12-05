import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load tensors from the eval_results folder
meta_score = torch.load(os.path.join("eval_results", "meta_score.pth"))
full_constraints_score = torch.load(
    os.path.join("eval_results", "always_active_score.pth")
)
no_constraints_score = torch.load(
    os.path.join("eval_results", "never_active_score.pth")
)
halfway_constraints_score = torch.load(
    os.path.join("eval_results", "halfway_score.pth")
)

# Concatenate tensors along the first dimension (axis 0)
score_tensor = torch.cat(
    [
        meta_score,
        full_constraints_score,
        no_constraints_score,
        halfway_constraints_score,
    ]
)
# Convert the score tensor to a list for DataFrame compatibility
score_list = score_tensor.tolist()

# Corresponding agent labels
agent_labels = (
    ["Meta Agent"] * len(meta_score)
    + ["Full Constraints"] * len(full_constraints_score)
    + ["No Constraints"] * len(no_constraints_score)
    + ["Halfway Constraints"] * len(halfway_constraints_score)
)

# Create DataFrame
data = pd.DataFrame({"Score": score_list, "Agent": agent_labels})

# Plot beeswarm plot
plt.figure(figsize=(10, 6))
sns.swarmplot(x="Score", y="Agent", data=data, size=8)
plt.title("True Return Distribution")
plt.xlabel("True Return")
plt.ylabel("Agent")
plt.show()
