import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

sns.set_style("whitegrid")

# Single argument, which is the path to the data file
parser = argparse.ArgumentParser()
parser.add_argument("data_file", type=str, help="Path to the data file")
args = parser.parse_args()

# Load the data
df = pd.read_csv(args.data_file)

# Parse the "returns" column from strings to lists of floats
df["returns"] = df["returns"].apply(lambda x: eval(x))  # Convert strings to lists of floats

# Expand the "returns" column into a long format
df_long = df.explode("returns").reset_index(drop=True)
df_long["returns"] = df_long["returns"].astype(float)

# sns.lineplot(
#     data=df,
#     x="percentage",
#     y="mean_return",
# )

# # Customize the plot
# plt.title("Mean return vs. time of constraint activation", fontsize=14)
# plt.xlabel("Constraint activated (% of training)", fontsize=12)
# plt.ylabel("Mean return", fontsize=12)
# plt.xticks(rotation=45, fontsize=10)  # Rotate x-axis labels if necessary
# plt.yticks(fontsize=10)
# plt.tight_layout()

# # Show the plot
# plt.show()

# Add an "index" column for x-axis
df_long["index"] = df_long.groupby(["percentage", "n_eval"]).cumcount()

# Plot using Seaborn
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df_long,
    x="index",
    y="returns",
    hue="percentage",
    style="percentage",
    markers=True,
    dashes=False,
)
plt.title("Errorline Plot: Returns for Each Percentage")
plt.xlabel("Evaluation Index")
plt.ylabel("Returns")
plt.grid(True)
plt.legend(title="Percentage")
plt.savefig(os.path.splitext(args.data_file)[0] + ".svg")
# plt.show()
