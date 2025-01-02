import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")

# Load the data
df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/percentage_influence.csv"))

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
