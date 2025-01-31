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


def get_trainer(env_config, agent_config, collector_config):
    env = ToyEnv.get_env(env_config)
    agent = ToyTabularQAgent(**agent_config)
    logger = ToyTabularQLogger(env, agent)

    trainer = OffpolicyTrainer(
        env=env,
        agent=agent,
        logger=logger,
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
    multiple_run_data = []
    for i in range(times_to_train):
        trainer = get_trainer(env_config, agent_config, collector_config)
        trainer.train(**trainer_kwargs)
        history = trainer.logger.dump()
        multiple_run_data.append(history)
    return multiple_run_data

def main():
    wandb.init(project="toy-base")

    env_config = {}
    env_config["n_states"] = 16
    env_config["shortcut_steps"] = 3
    env_config["big_reward"] = 10
    env_config["punishment"] = 0.0
    env_config["gamma"] = 0.99
    env_config["device"] = torch.device("cpu")
    env_config["return_x"] = 5
    env_config["return_y"] = 1
    env_config["max_steps"] = 10*env_config["n_states"]

    collector_config = {}
    collector_config["batch_size"] = 64
    collector_config["total_frames"] = 100_000
    collector_config["random_frames"] = 0
    collector_config["device"] = env_config["device"]

    agent_config = {}
    agent_config["n_states"] = env_config["n_states"]
    agent_config["n_actions"] = 4
    agent_config["gamma"] = env_config["gamma"]
    agent_config["lr"] = 1e-1
    agent_config["epsilon"] = 0.1
    agent_config["replay_buffer_size"] = collector_config["total_frames"]
    agent_config["device"] = env_config["device"]
    agent_config["rb_alpha"] = 0.7
    agent_config["rb_beta"] = 0.5
    agent_config["rb_batch_size"] = 64
    agent_config["num_optim_steps"] = 1
    agent_config["init_qvalues"] = 10

    unconstrained_multiple_run_data = multiple_runs(times_to_train=5, env_config=env_config, agent_config=agent_config, collector_config=collector_config, when_constraints_active=1.0, times_to_eval=10)
    constrained_multiple_run_data = multiple_runs(times_to_train=5, env_config=env_config, agent_config=agent_config, collector_config=collector_config, when_constraints_active=0.0, times_to_eval=10)
    toggled_multiple_run_data = multiple_runs(times_to_train=5, env_config=env_config, agent_config=agent_config, collector_config=collector_config, when_constraints_active=0.5, times_to_eval=10)

    print("hello?")


    # Pickle the data
    print("Pickling data...", end="")
    with open("data/2025-01-31|16-states.pkl", "wb") as f:
        pickle.dump({
            "multiple_run_data": {
                "unconstrained": unconstrained_multiple_run_data,
                "constrained": constrained_multiple_run_data,
                "toggled": toggled_multiple_run_data
            },
            "env_config": env_config,
        }, f)
    print("Done")


if __name__ == "__main__":
    main()
