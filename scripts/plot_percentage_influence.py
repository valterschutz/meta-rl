import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

sns.set_style("whitegrid")

# Single argument, which is the path to the data file
parser = argparse.ArgumentParser()
parser.add_argument("data_file", type=str, help="Path to the data file")
parser.add_argument("--save", action="store_true", help="Save the plot")
args = parser.parse_args()

# Load the data
df = pd.read_csv(args.data_file)

# Parse the "returns" column from strings to lists of floats
df["returns"] = df["returns"].apply(lambda x: eval(x))  # Convert strings to lists of floats

# Expand the "returns" column into a long format
df_long = df.explode("returns").reset_index(drop=True)
df_long["returns"] = df_long["returns"].astype(float)

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
    ci=None  # Disable error bars
)
plt.title("Mean Plot: Returns for Each Percentage")
plt.xlabel("Evaluation Index")
plt.ylabel("Returns")
plt.grid(True)
plt.legend(title="Percentage")

if args.save:
    plt.savefig(os.path.splitext(args.data_file)[0] + ".svg")
else:
    plt.show()
