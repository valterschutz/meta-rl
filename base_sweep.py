import wandb

from train_base import train_base

wandb.login()


def main():
    wandb.init(project="base-sweep")
    return_dissimilarity = train_base(wandb.config)
    wandb.log({"return_dissimilarity": return_dissimilarity})


# 2: Define the search space
sweep_configuration = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "return_dissimilarity"},
    "parameters": {
        "n_states": {"values": [20]},
        "return_x": {"values": [0.2]},
        "return_y": {"values": [0.1]},
        "big_reward": {"values": [10.0]},
        "gamma": {"values": [0.99]},
        "punishment": {"values": [0.0]},
        "device": {"values": ["cpu"]},
        "num_optim_epochs": {"min": 1, "max": 20},
        "batch_size": {"min": 1, "max": 100},
        "max_grad_norm": {"values": 1},
        "lmbda": {"min": 0.1, "max": 0.99},
        "total_frames": {"values": [1000]},
        "rollout_timeout": {"values": [200]},
        "lr": {"min": 1e-4, "max": 1e-1},
        "constraints_enabled": {"values": [True]},
    },
}

# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project="base-sweep")

# TODO: run overnight
wandb.agent(sweep_id, function=main, count=1000)
