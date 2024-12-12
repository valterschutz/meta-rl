import json
import pickle
import sys
from datetime import datetime
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torchrl.envs.utils import (
    ExplorationType,
    check_env_specs,
    set_exploration_type,
    step_mdp,
)
from torchrl.collectors import SyncDataCollector
from tqdm import tqdm

import wandb
from agents import MetaAgent
from base import get_base_from_config
from env import MetaEnv
from utils import DictWrapper, MethodLogger
import argparse
import tensordict
import yaml

from tensordict import TensorDict


def plot_to_pil(data, title, xlabel, ylabel, xticklabels, yticklabels):
    # Create the plot
    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cbar = fig.colorbar(cax, ax=ax)
    # cbar.set_label("Value")
    ax.set_xticks(range(len(xticklabels)))
    ax.set_xticklabels([f"{x:.1f}" for x in xticklabels], rotation=45, ha="right")
    ax.set_yticks(range(len(yticklabels)))
    ax.set_yticklabels([f"{y:.1f}" for y in yticklabels])
    plt.tight_layout()

    # Convert the plot to a PIL image
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)


def plot_vector_to_pil(data, title, label, ticklabels):
    """
    Create a plot from vector data and convert it to a PIL image.

    Args:
        data (numpy.ndarray): The data to plot.
        title (str): The title of the plot.
        label (str): The label for the x-axis.
        ticklabels (list): The tick labels for the x-axis.

    Returns:
        PIL.Image: The plot as a PIL image.
    """
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 4))
    cax = ax.imshow(data.reshape(1, -1), aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel(label)
    fig.colorbar(cax, ax=ax, orientation="horizontal", pad=0.2)
    ax.set_xticks(range(len(ticklabels)))
    ax.set_xticklabels([f"{x:.1f}" for x in ticklabels])
    ax.set_yticks([])
    plt.tight_layout(pad=4.0)

    # Convert the plot to a PIL image
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)


def prep_data(steps_per_episode):
    action_ticks = torch.linspace(0, 1, 11)
    step_ticks = range(0, steps_per_episode + 1)
    action_samples_meshed, step_samples_meshed = torch.meshgrid(
        action_ticks, torch.tensor(step_ticks, dtype=torch.float32)
    )
    # action_samples = action_ticks.unsqueeze(-1)
    step_samples = torch.tensor(step_ticks, dtype=torch.float32).unsqueeze(-1)
    action_ticks = action_ticks.tolist()
    step_ticks = list(step_ticks)
    action_samples_meshed = action_samples_meshed.unsqueeze(-1)
    step_samples_meshed = step_samples_meshed.unsqueeze(-1)
    policy_td = TensorDict(
        {
            "step": step_samples,
        },
        batch_size=(steps_per_episode + 1,),
    )
    qvalue_td = TensorDict(
        {
            "step": step_samples_meshed,
            "action": action_samples_meshed,
        },
        batch_size=(11, steps_per_episode + 1),
    )

    return policy_td, qvalue_td, action_ticks, step_ticks


def calc_ssd(old_params, new_params):
    ssd = 0.0
    for old, new in zip(old_params.values(), new_params.values()):
        ssd += torch.sum((old - new) ** 2).item()
    return ssd


def get_params(module):
    return {name: param.clone() for name, param in module.named_parameters()}


parser = argparse.ArgumentParser()
parser.add_argument("base_config", type=str)
parser.add_argument("meta_config", type=str)
args = parser.parse_args()
with open(args.base_config, "r", encoding="UTF-8") as f:
    base_config = yaml.safe_load(f)
with open(args.meta_config, "r", encoding="UTF-8") as f:
    meta_config = yaml.safe_load(f)
meta_steps_per_episode = base_config["total_frames"] // base_config["batch_size"]
meta_total_steps = meta_steps_per_episode * meta_config["train_episodes"]
print(f"Meta steps per episode: {meta_steps_per_episode}")

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
meta_agent = PPOMetaAgent(
    state_spec=meta_env.state_spec,
    action_spec=meta_env.action_spec,
    num_optim_epochs=meta_config["num_optim_epochs"],
    buffer_size=base_config["batch_size"],
    device=meta_config["device"],
    max_policy_grad_norm=meta_config["max_policy_grad_norm"],
    max_value_grad_norm=meta_config["max_value_grad_norm"],
    lr=meta_config["lr"],
    gamma=meta_config["gamma"],
    hidden_units=meta_config["hidden_units"],
    clip_epsilon=meta_config["clip_epsilon"],
    use_entropy=meta_config["use_entropy"],
    entropy_coef=meta_config["entropy_coef"],
    critic_coef=meta_config["critic_coef"],
)

pbar = tqdm(total=meta_total_steps)

wandb.login()
wandb.init(
    project="toy-meta-train",
    name=f"toy-meta-train|{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    config={
        **{f"meta-{k}": v for k, v in meta_config.items()},
        **{f"base-{k}": v for k, v in base_config.items()},
    },
)

try:
    # For visualizing policy and qvalues
    # step_td, step_action_td, action_ticks, step_ticks = prep_data(
    #     meta_steps_per_episode
    # )
    # For logging SSD of params during training
    old_value_params = get_params(meta_agent.value_module)
    old_policy_params = get_params(meta_agent.policy_module)

    for i in range(meta_config["train_episodes"]):
        meta_td = meta_env.reset()  # Resets base agent in meta environment
        for j in range(meta_steps_per_episode):
            meta_td = meta_agent.policy(meta_td)
            meta_td = meta_env.step(meta_td)
            meta_losses, meta_avg_policy_grad_norm, meta_avg_value_grad_norm = (
                meta_agent.process_batch(meta_td.unsqueeze(0))
            )
            pbar.update(meta_td.numel())

            if (i * meta_steps_per_episode + j) % meta_config[
                "expensive_log_interval"
            ] == 0:
                value_params = get_params(meta_agent.value_module)
                policy_params = get_params(meta_agent.policy_module)
                value_params_ssd = calc_ssd(old_value_params, value_params)
                policy_params_ssd = calc_ssd(old_policy_params, policy_params)
                old_value_params = get_params(meta_agent.value_module)
                old_policy_params = get_params(meta_agent.policy_module)
                # Visualize policy probabilities and Q-values
                # policy_td = meta_agent.policy_module(step_td)
                # qvalue_td = meta_agent.qvalue_module(step_action_td)
                log_dict0 = {
                    # "policy": wandb.Image(
                    #     plot_vector_to_pil(
                    #         policy_td["action"].detach().cpu().numpy(),
                    #         "Policy",
                    #         "Step",
                    #         ticklabels=step_ticks,
                    #     )
                    # ),
                    # "Q-values": wandb.Image(
                    #     plot_to_pil(
                    #         qvalue_td["state_action_value"].detach().cpu().numpy(),
                    #         "Q-values",
                    #         "Step",
                    #         "Action",
                    #         xticklabels=step_ticks,
                    #         yticklabels=action_ticks,
                    #     )
                    # ),
                    "Value params SSD": value_params_ssd,
                    "Policy params SSD": policy_params_ssd,
                }
            else:
                log_dict0 = {}

            log_dict1 = {
                "step": j,
                "base_true_mean_reward": meta_td["base_true_mean_reward"].item(),
                "base_mean_reward": meta_td["base_mean_reward"].item(),
                "base_std_reward": meta_td["base_std_reward"].item(),
                "last_action": meta_td["last_action"].item(),
                "action (sampled)": meta_td["action"].item(),
                "action (mean)": meta_td["loc"].item(),
                "action (variance)": meta_td["scale"].item(),
                "meta reward": meta_td["next", "reward"].item(),
                "base loss_objective": meta_td[
                    "base", "losses", "loss_objective"
                ].item(),
                "base loss_critic": meta_td["base", "losses", "loss_critic"].item(),
                "base loss_entropy": meta_td["base", "losses", "loss_entropy"].item(),
                "base state distribution": wandb.Histogram(
                    meta_td["base", "states"].argmax(dim=-1)
                ),
                "base reward distribution": wandb.Histogram(meta_td["base", "rewards"]),
                "base true_reward distribution": wandb.Histogram(
                    meta_td["base", "true_rewards"]
                ),
            }
            if meta_losses is not None:
                log_dict2 = {
                    "meta loss_objective": meta_losses["loss_objective"].item(),
                    "meta loss_critic": meta_losses["loss_critic"].item(),
                    "meta loss_entropy": meta_losses["loss_entropy"].item(),
                    "meta avg_policy_grad_norm": meta_avg_policy_grad_norm,
                    "meta avg_value_grad_norm": meta_avg_value_grad_norm,
                }
            else:
                log_dict2 = {}

            wandb.log({**log_dict0, **log_dict1, **log_dict2})

            meta_td = step_mdp(meta_td)
except KeyboardInterrupt:
    print("Training interrupted.")

# Save meta agent
print(f"Saving policy module to models/{meta_config['policy_module_name']}.pth")
torch.save(
    meta_agent.policy_module.state_dict(),
    f"models/{meta_config['policy_module_name']}.pth",
)
print(f"Saving value module to models/{meta_config['value_module_name']}.pth")
torch.save(
    meta_agent.value_module.state_dict(),
    f"models/{meta_config['value_module_name']}.pth",
)
