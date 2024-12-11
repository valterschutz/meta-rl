import argparse
import yaml
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


def no_constraints_policy(td):
    td["action"] = torch.tensor([0.0], dtype=torch.float32, device=td.device)
    return td


def full_constraints_policy(td):
    td["action"] = torch.tensor([1.0], dtype=torch.float32, device=td.device)
    return td


def halfway_constraints_policy(td):
    if td["step"] < meta_steps_per_episode // 2:
        td["action"] = torch.tensor([0.0], dtype=torch.float32, device=td.device)
    else:
        td["action"] = torch.tensor([1.0], dtype=torch.float32, device=td.device)
    return td


def random_policy(td):
    pass


def loss_based_policy(td):
    pass


def eval_meta_policy(meta_env, meta_config, base_config, meta_policy, verbose=False):
    pbar = tqdm(total=meta_config["eval_episodes"])
    score = torch.zeros(meta_config["eval_episodes"], dtype=torch.float32)
    for i in range(meta_config["eval_episodes"]):
        # Train the base agent using some meta policy
        _ = meta_env.rollout(meta_config["rollout_timeout"], meta_policy)
        # Once base agent is trained, evaluate it
        with set_exploration_type(ExplorationType.DETERMINISTIC):
            base_td = meta_env.base_env.rollout(
                base_config["rollout_timeout"], base_agent.policy
            )
        # If verbose, print out all the true rewards
        if verbose:
            print(base_td["next", "true_reward"])
        # Calculate true return experienced by base agent
        score[i] = calc_return(
            base_td["next", "true_reward"].flatten(), base_config["gamma"]
        )
        pbar.update(1)
    return score


def evaluate_and_save(policy_name, policy_fn):
    print(f"Evaluating base agent with {policy_name}...")
    score = eval_meta_policy(meta_env, meta_config, base_config, policy_fn)
    path = f"eval_results/{policy_name}_score.pth"
    torch.save(score, path)
    print(f"  Results saved to {path}")


parser = argparse.ArgumentParser()
# Config for how to train the base agent
parser.add_argument("base_config", type=str)
# Config for how to initialize the meta agent (a bit unnecessary)
parser.add_argument("meta_config", type=str)
# Config for which baselines to evaluate
parser.add_argument("eval_config", type=str)
args = parser.parse_args()
with open(args.base_config, "r", encoding="UTF-8") as f:
    base_config = yaml.safe_load(f)
with open(args.meta_config, "r", encoding="UTF-8") as f:
    meta_config = yaml.safe_load(f)
with open(args.eval_config, "r", encoding="UTF-8") as f:
    eval_config = yaml.safe_load(f)


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
    num_optim_epochs=meta_config["num_optim_epochs"],
    buffer_size=meta_config["buffer_size"],
    sub_batch_size=meta_config["sub_batch_size"],
    device=meta_config["device"],
    max_policy_grad_norm=meta_config["max_policy_grad_norm"],
    max_qvalue_grad_norm=meta_config["max_qvalue_grad_norm"],
    policy_lr=meta_config["policy_lr"],
    qvalue_lr=meta_config["qvalue_lr"],
    gamma=meta_config["gamma"],
    hidden_units=meta_config["hidden_units"],
    target_eps=meta_config["target_eps"],
    replay_alpha=meta_config["replay_alpha"],
    replay_beta=meta_config["replay_beta"],
)

# Load saved meta agent
meta_agent.reset(
    mode="eval",
    policy_module_state_dict=torch.load(
        f"models/{meta_config['policy_module_name']}.pth"
    ),
    qvalue_module_state_dict=torch.load(
        f"models/{meta_config['qvalue_module_name']}.pth"
    ),
)

meta_steps_per_episode = base_config["total_frames"] // base_config["batch_size"]

if eval_config["meta"]:
    evaluate_and_save("meta", meta_agent.policy)

# if eval_config["random_policy"]:
#     evaluate_and_save("random", random_policy)

# if eval_config["loss_based_policy"]:
#     evaluate_and_save("loss_based", loss_based_policy)

if eval_config["never_active"]:
    evaluate_and_save("never_active", no_constraints_policy)

if eval_config["always_active"]:
    evaluate_and_save("always_active", full_constraints_policy)

if eval_config["halfway"]:
    evaluate_and_save("halfway", halfway_constraints_policy)
