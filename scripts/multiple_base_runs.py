import os
import sys
import pickle

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

import torch
import numpy as np
import pandas as pd

import wandb
from trainers import OffpolicyTrainer
from agents.base_agents import ToyTabularDQNAgent
from envs.toy_env import ToyEnv

from torchrl.envs.transforms import Compose, StepCounter, DoubleToFloat, TransformedEnv, DTypeCastTransform


def get_env(env_config):

    x, y = ToyEnv.calculate_xy(n_states=env_config["n_states"], shortcut_steps=env_config["shortcut_steps"], return_x=env_config["return_x"], return_y=env_config["return_y"], big_reward=env_config["big_reward"], punishment=0, gamma=env_config["env_gamma"])

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

    env = TransformedEnv(
        env,
        Compose(
            StepCounter(max_steps=env_config["max_steps"]),
            # DTypeCastTransform(dtype_in=torch.long, dtype_out=torch.float32, in_keys=["observation"]),
        )
    )

    return env


def get_agent(env, agent_config, env_config, collector_config):
    agent = ToyTabularDQNAgent(
        device=agent_config["device"],
        batch_size=collector_config["batch_size"],
        sub_batch_size=collector_config["batch_size"],
        num_epochs=1,
        replay_buffer_args={
            "buffer_size": collector_config["total_frames"],
            "min_buffer_size": agent_config["min_buffer_size"],
            "alpha": agent_config["alpha"],
            "beta": agent_config["beta"],
        },
        env=env,
        agent_detail_args={
            "agent_gamma": agent_config["agent_gamma"],
            "target_eps": agent_config["target_eps"],
            "value_lr": agent_config["value_lr"],
            "value_max_grad": agent_config["value_max_grad"],
            "n_states": env_config["n_states"],
            "qvalue_eps": agent_config["qvalue_eps"],
        }
    )

    return agent

def get_trainer(env_config, agent_config, collector_config):
    env = get_env(env_config)
    agent = get_agent(env, agent_config, env_config, collector_config)
    trainer = OffpolicyTrainer(
        env=env,
        agent=agent,
        progress_bar=True,
        times_to_eval=2,
        collector_device=collector_config["device"],
        log=True,
        max_eval_steps=env_config["max_steps"],
        collector_args={
            "batch_size": collector_config["batch_size"],
            "total_frames": collector_config["total_frames"],
        },
        env_gamma=env_config["env_gamma"],
        eval_env=None
    )
    return trainer

def multiple_runs(times_to_train, env_config, agent_config, collector_config, **trainer_kwargs):
    result_dicts = []
    for i in range(times_to_train):
        trainer = get_trainer(env_config, agent_config, collector_config)
        result_dict = trainer.train(**trainer_kwargs)
        result_dicts.append(result_dict)
    return result_dicts

def main():
    wandb.init(project="toy-base")

    env_config = {}
    env_config["n_states"] = 10
    env_config["device"] = torch.device("cpu")
    env_config["env_gamma"] = 0.999
    env_config["max_steps"] = 5*env_config["n_states"]
    env_config["shortcut_steps"] = 2
    env_config["big_reward"] = 10
    env_config["return_x"] = 5
    env_config["return_y"] = 1

    collector_config = {}
    collector_config["batch_size"] = 64
    collector_config["total_frames"] = 200_000
    collector_config["device"] = env_config["device"]

    agent_config = {}
    agent_config["agent_gamma"] = 0.999
    agent_config["buffer_size"] = collector_config["total_frames"]
    agent_config["min_buffer_size"] = 0
    agent_config["alpha"] = 0.7
    agent_config["beta"] = 0.5
    agent_config["device"] = env_config["device"]

    agent_config["target_eps"] = 0.99
    agent_config["value_lr"] = 1e-2
    agent_config["value_max_grad"] = 1
    agent_config["qvalue_eps"] = 0.05

    unconstrained_result_dicts = multiple_runs(times_to_train=5, env_config=env_config, agent_config=agent_config, collector_config=collector_config, when_constraints_active=1.0)
    constrained_result_dicts = multiple_runs(times_to_train=5, env_config=env_config, agent_config=agent_config, collector_config=collector_config, when_constraints_active=0.0)

    # Pickle the data
    with open("data/unconstrained_result_dicts|2025-01-29|10-states.pkl", "wb") as f:
        pickle.dump(unconstrained_result_dicts, f)
    with open("data/constrained_result_dicts|2025-01-29|10-states.pkl", "wb") as f:
        pickle.dump(constrained_result_dicts, f)


if __name__ == "__main__":
    main()
