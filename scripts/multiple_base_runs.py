import os
import sys
import pickle

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

import torch
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
import pandas as pd

import wandb
from trainers import OffpolicyTrainer
from loggers import ToyTabularQLogger
from agents.base_agents import ToyTabularQAgent
from envs.toy_env import ToyEnv

from torchrl.envs.transforms import Compose, StepCounter, DoubleToFloat, TransformedEnv, DTypeCastTransform


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

    env = TransformedEnv(
        env,
        Compose(
            StepCounter(max_steps=env_config["max_steps"]),
            # DTypeCastTransform(dtype_in=torch.long, dtype_out=torch.float32, in_keys=["observation"]),
        )
    )

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

def get_trainer(env_config, agent_config, collector_config):
    env = get_env(env_config)
    agent = get_agent(env, agent_config, env_config, collector_config)

    trainer = OffpolicyTrainer(
        env=env,
        agent=agent,
        logger=ToyTabularQLogger(env, agent),
        progress_bar=True,
        max_eval_steps=env_config["max_steps"],
        collector_args={
            "batch_size": collector_config["batch_size"],
            "total_frames": collector_config["total_frames"],
            "random_frames": collector_config["random_frames"],
            "device": collector_config["device"],
        },
        eval_env=None
    )
    return trainer

def multiple_runs(times_to_train, env_config, agent_config, collector_config, **trainer_kwargs):
    result_dicts = []
    for i in range(times_to_train):
        trainer = get_trainer(env_config, agent_config, collector_config)
        trainer.train(**trainer_kwargs)
        result_dicts.append(trainer.logger.dump())
    return result_dicts

def main():
    wandb.init(project="toy-base")

    env_config = {}
    env_config["n_states"] = 16
    env_config["device"] = torch.device("cpu")
    env_config["env_gamma"] = 0.99
    env_config["max_steps"] = 10*env_config["n_states"]
    env_config["shortcut_steps"] = 3
    env_config["big_reward"] = 10
    env_config["return_x"] = 5
    env_config["return_y"] = 1

    collector_config = {}
    collector_config["batch_size"] = 64
    collector_config["total_frames"] = 500_000
    collector_config["random_frames"] = 0
    collector_config["device"] = env_config["device"]

    agent_config = {}
    agent_config["agent_gamma"] = env_config["env_gamma"]
    agent_config["lr"] = 1e-2
    agent_config["epsilon"] = 0.5
    agent_config["replay_buffer_size"] = 1000
    agent_config["device"] = env_config["device"]
    agent_config["rb_alpha"] = 0.7
    agent_config["rb_beta"] = 0.5
    agent_config["rb_batch_size"] = 64
    agent_config["num_optim_steps"] = 1

    # unconstrained_result_dicts = multiple_runs(times_to_train=1, env_config=env_config, agent_config=agent_config, collector_config=collector_config, when_constraints_active=1.0, times_to_eval=10)
    constrained_result_dicts = multiple_runs(times_to_train=1, env_config=env_config, agent_config=agent_config, collector_config=collector_config, when_constraints_active=0.0, times_to_eval=10)


    # Pickle the data
    # with open("data/unconstrained_result_dicts|2025-01-29|10-states.pkl", "wb") as f:
    #     pickle.dump(unconstrained_result_dicts, f)
    # with open("data/constrained_result_dicts|2025-01-29|10-states.pkl", "wb") as f:
    #     pickle.dump(constrained_result_dicts, f)


if __name__ == "__main__":
    main()
