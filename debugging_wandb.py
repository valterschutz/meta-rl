import wandb
from datetime import datetime

wandb.login()
wandb.init(
    project="debugging",
    name=f"debugging|{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    config={},
)

for i in range(42):
    wandb.log({"placeholder": i})
