from tensordict import TensorDict
import matplotlib.pyplot as plt
import os
import sys
import pickle

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

import torch
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
import pandas as pd

import wandb
from trainers import OffpolicyTrainer
from loggers import ToyTabularQLogger
from agents.base_agents import ToyTabularQAgent
from envs.toy_env import ToyEnv

from torchrl.envs.transforms import Compose, StepCounter, DoubleToFloat, TransformedEnv, DTypeCastTransform

# Figure out why TabularQAgent is not learning properly

def get_env(env_config):

    x, y = ToyEnv.calculate_xy(n_states=env_config["n_states"], shortcut_steps=env_config["shortcut_steps"], return_x=env_config["return_x"], return_y=env_config["return_y"], big_reward=env_config["big_reward"], gamma=env_config["env_gamma"])

    env = ToyEnv(
        left_reward=x,
        right_reward=x,
        down_reward=y,
        up_reward=y,
        n_states=env_config["n_states"],
        shortcut_steps=env_config["shortcut_steps"],
        big_reward=env_config["big_reward"],
        punishment=0.0,
        gamma=env_config["env_gamma"],
        constraints_active=False,
        random_start=False,
        seed=None,
        device=env_config["device"])

    # env = TransformedEnv(
    #     env,
    #     Compose(
    #         StepCounter(max_steps=env_config["max_steps"]),
    #         # DTypeCastTransform(dtype_in=torch.long, dtype_out=torch.float32, in_keys=["observation"]),
    #     )
    # )

    return env


def get_agent(env, agent_config, env_config, collector_config):
    agent = ToyTabularQAgent(
        n_states=env_config["n_states"],
        agent_gamma=agent_config["agent_gamma"],
        lr=agent_config["lr"],
        epsilon=agent_config["epsilon"],
        replay_buffer_size=agent_config["replay_buffer_size"],
        device=agent_config["device"],
        rb_alpha=agent_config["rb_alpha"],
        rb_beta=agent_config["rb_beta"],
        rb_batch_size=agent_config["rb_batch_size"],
        num_optim_steps=agent_config["num_optim_steps"]
    )


    return agent

def main():
    env_config = {}
    env_config["n_states"] = 16
    env_config["device"] = torch.device("cpu")
    env_config["env_gamma"] = 0.99
    env_config["max_steps"] = 10*env_config["n_states"]
    env_config["shortcut_steps"] = 3
    env_config["big_reward"] = 10
    env_config["return_x"] = 5
    env_config["return_y"] = 1

    agent_config = {}
    agent_config["agent_gamma"] = env_config["env_gamma"]
    agent_config["lr"] = 1e-2
    agent_config["epsilon"] = 0.1
    agent_config["replay_buffer_size"] = 1000
    agent_config["device"] = env_config["device"]
    agent_config["rb_alpha"] = 0.7
    agent_config["rb_beta"] = 0.5
    agent_config["rb_batch_size"] = 64
    agent_config["num_optim_steps"] = 1

    env = get_env(env_config)
    agent = get_agent(env, agent_config, env_config, None)

    # Produce a tensordict that represents the whole dataset
    # States go from 0 to n_states-1 and actions from 0 to 3
    n_states = env_config["n_states"]
    n_actions = 4
    states = torch.arange(env_config["n_states"]).repeat(n_actions)
    actions = torch.arange(n_actions).repeat_interleave(n_states)
    td = TensorDict({
        "observation": states.unsqueeze(-1),
        "action": F.one_hot(actions, num_classes=4),
    }, batch_size=(n_states*n_actions,))
    td = env.step(td)

    optimal_qvalues = env.calc_optimal_qvalues()
    errors = []
    for i in range(1000):
        agent.process_batch(td, constraints_active=False)
        errors.append((agent.qvalues - optimal_qvalues).abs().sum().item())
    plt.plot(errors)
    plt.show()


if __name__ == "__main__":
    main()
