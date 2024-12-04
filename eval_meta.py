import argparse
import json
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensordict
import torch
from torchrl.envs.utils import (
    ExplorationType,
    check_env_specs,
    set_exploration_type,
    step_mdp,
)
from tqdm import tqdm

import wandb
from agents import MetaAgent
from base import get_base_from_config
from env import MetaEnv
from utils import DictWrapper, MethodLogger, calc_return


def eval_meta_policy(meta_env, meta_config, base_config, meta_policy):
    pbar = tqdm(total=meta_config["eval_episodes"])
    score = torch.zeros(meta_config["eval_episodes"], dtype=torch.float32)
    for i in range(meta_config["eval_episodes"]):
        meta_td = meta_env.rollout(meta_config["rollout_timeout"], meta_policy)
        # Calculate return experienced by base agent
        episode_base_true_return = 0
        for j in range(len(meta_td)):
            episode_base_true_return += calc_return(
                meta_td["base", "true_rewards"][j],
                base_config["gamma"],
                discount_start=j * base_config["batch_size"],
            )
        score[i] = episode_base_true_return
        pbar.update(1)
    return score


parser = argparse.ArgumentParser()
parser.add_argument("base_config", type=str)
parser.add_argument("meta_config", type=str)
args = parser.parse_args()
with open(args.base_config, "r", encoding="UTF-8") as f:
    base_config = json.load(f)
with open(args.meta_config, "r", encoding="UTF-8") as f:
    meta_config = json.load(f)

base_env, base_agent, base_collector_fn = get_base_from_config(DictWrapper(base_config))

# Meta env
meta_env = MetaEnv(
    base_env=base_env,
    base_agent=base_agent,
    base_collector_fn=base_collector_fn,
    device=meta_config["device"],
)

check_env_specs(meta_env)

# Meta agent
meta_agent = MetaAgent(
    state_spec=meta_env.state_spec,
    action_spec=meta_env.action_spec,
    num_optim_epochs=1,
    buffer_size=1,
    sub_batch_size=1,
    device=meta_config["device"],
    max_grad_norm=meta_config["max_grad_norm"],
    lr=meta_config["lr"],
    gamma=meta_config["gamma"],
    lmbda=meta_config["lmbda"],
    clip_epsilon=meta_config["clip_epsilon"],
    use_entropy=meta_config["use_entropy"],
    hidden_units=meta_config["hidden_units"],
)

# Load saved meta agent
meta_agent.reset(
    mode="eval",
    policy_module_state_dict=torch.load(
        f"models/{meta_config['policy_module_name']}.pth"
    ),
    value_module_state_dict=torch.load(
        f"models/{meta_config['value_module_name']}.pth"
    ),
)

meta_steps_per_episode = base_config["total_frames"] // base_config["batch_size"]

print(f"Evaluating base agent with constraints selected by meta agent...")
meta_score = eval_meta_policy(meta_env, meta_config, base_config, meta_agent.policy)
print(f"{len(meta_score)=}")


def full_constraints_policy(td):
    td["action"] = torch.tensor([1.0], dtype=torch.float32, device=td.device)
    return td


print(f"Evaluating base agent with constraints fully active...")
full_constraints_score = eval_meta_policy(
    meta_env, meta_config, base_config, full_constraints_policy
)
print(f"{len(full_constraints_score)=}")


def no_constraints_policy(td):
    td["action"] = torch.tensor([0.0], dtype=torch.float32, device=td.device)
    return td


print(f"Evaluating base agent with constraints disabled...")
no_constraints_score = eval_meta_policy(
    meta_env, meta_config, base_config, no_constraints_policy
)
print(f"{len(no_constraints_score)=}")


def halfway_constraints_policy(td):
    if td["step"] < meta_steps_per_episode // 2:
        td["action"] = torch.tensor([0.0], dtype=torch.float32, device=td.device)
    else:
        td["action"] = torch.tensor([1.0], dtype=torch.float32, device=td.device)
    return td


print(f"Evaluating base agent with constraints activated at halfway...")
halfway_constraints_score = eval_meta_policy(
    meta_env, meta_config, base_config, halfway_constraints_policy
)
print(f"{len(halfway_constraints_score)=}")

# Check the length of each list again before concatenating
print("Meta Agent Length:", len(meta_score))
print("Full Constraints Length:", len(full_constraints_score))
print("No Constraints Length:", len(no_constraints_score))
print("Halfway Constraints Length:", len(halfway_constraints_score))

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
