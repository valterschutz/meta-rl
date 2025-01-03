import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io

# Generate sample data
np.random.seed(42)  # For reproducibility
data = "percentage,n_eval,returns\n"
percentages = [0, 25, 50, 75, 100]
n_evals = [0, 1, 2, 3]

for percentage in percentages:
    for n_eval in n_evals:
        returns = np.random.normal(loc=percentage / 100, scale=0.1, size=10)  # Random returns
        data += f"{percentage},{n_eval},\"{list(returns)}\"\n"

# Read the data into a pandas DataFrame
df = pd.read_csv(io.StringIO(data))

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
)
plt.title("Errorline Plot: Returns for Each Percentage")
plt.xlabel("Evaluation Index")
plt.ylabel("Returns")
plt.grid(True)
plt.legend(title="Percentage")
plt.show()
