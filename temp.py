from datetime import datetime, timezone
import wandb


wandb.init(
    project="temp",
    name=f"temp|{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')}",
    config={},
)
