import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load two saved dataframes
unconstrained_qvalues = pd.read_csv("data/unconstrained_qvalues|2025-01-27|30-states.csv")
constrained_qvalues = pd.read_csv("data/constrained_qvalues|2025-01-27|30-states.csv")

unconstrained_qvalues["type"] = "unconstrained"
constrained_qvalues["type"] = "constrained"

# Merge the two dataframes
df = pd.concat(
    [unconstrained_qvalues, constrained_qvalues],
    ignore_index=True
)

print(df)


sns.lineplot(
    data=df,
    x="batch",
    y="qvalue",
    hue="type",
)
plt.show()
