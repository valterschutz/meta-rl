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

sns.lineplot(
    data=df,
    x="percentage",
    y="mean_return",
)

# Customize the plot
plt.title("Mean return vs. time of constraint activation", fontsize=14)
plt.xlabel("Constraint activated (% of training)", fontsize=12)
plt.ylabel("Mean return", fontsize=12)
plt.xticks(rotation=45, fontsize=10)  # Rotate x-axis labels if necessary
plt.yticks(fontsize=10)
plt.tight_layout()

# Show the plot
plt.show()
