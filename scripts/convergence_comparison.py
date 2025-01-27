import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

import torch
import numpy as np
import pandas as pd

import wandb
from trainers import OffpolicyTrainer
from agents.base_agents import ToyTabularDQNAgent
from envs.toy_env import ToyEnv

from torchrl.envs.transforms import Compose, StepCounter, DoubleToFloat, TransformedEnv, DTypeCastTransform


def get_env(config):

    x, y = ToyEnv.calculate_xy(n_states=config["n_states"], shortcut_steps=config["shortcut_steps"], return_x=config["return_x"], return_y=config["return_y"], big_reward=config["big_reward"], punishment=0, gamma=config["env_gamma"])

    env = ToyEnv(
        left_reward=x,
        right_reward=x,
        down_reward=y,
        up_reward=y,
        n_states=config["n_states"],
        shortcut_steps=config["shortcut_steps"],
        big_reward=config["big_reward"],
        punishment=0.0,
        constraints_active=False,
        random_start=False,
        seed=None,
        device=config["device"])

    env = TransformedEnv(
        env,
        Compose(
            StepCounter(max_steps=config["max_steps"]),
            # DTypeCastTransform(dtype_in=torch.long, dtype_out=torch.float32, in_keys=["observation"]),
        )
    )

    return env


def get_agent(env, config):
    agent = ToyTabularDQNAgent(
        device=config["device"],
        batch_size=config["batch_size"],
        sub_batch_size=config["batch_size"],
        num_epochs=1,
        replay_buffer_args={
            "buffer_size": config["total_frames"],
            "min_buffer_size": 0,
            "alpha": 0.7,
            "beta": 0.5
        },
        env=env,
        agent_detail_args={
            "agent_gamma": config["agent_gamma"],
            "target_eps": 0.99,
            "value_lr": 1e-1,
            "value_max_grad": 1,
            "n_states": config["n_states"],
            "qvalue_eps": 0.05
        }
    )

    return agent

def get_trainer(config):
    env = get_env(config)
    agent = get_agent(env, config)
    trainer = OffpolicyTrainer(
        env=env,
        agent=agent,
        progress_bar=True,
        times_to_eval=2,
        collector_device=config["device"],
        log=True,
        max_eval_steps=config["max_steps"],
        collector_args={
            "batch_size": config["batch_size"],
            "total_frames": config["total_frames"],
        },
        env_gamma=config["env_gamma"],
        eval_env=None
    )
    return trainer

def get_qvalues(when_constraints_active, times_to_train, config):
    qvalues = []
    for i in range(times_to_train):
        trainer = get_trainer(config)
        eval_true_returns, train_info_dicts = trainer.train(when_constraints_active=when_constraints_active)
        q_values = []
        for d in train_info_dicts:
            q_values.append(d["qvalues"])
        qvalues.append(np.array(q_values))
    qvalues = np.array(qvalues)
    qvalues = pd.DataFrame(qvalues)
    qvalues["run"] = qvalues.index
    qvalues = qvalues.melt(id_vars="run", var_name="batch", value_name="qvalue")
    return qvalues

def main():
    wandb.init(project="toy-base")

    config = {}
    config["n_states"] = 50
    config["device"] = torch.device("cpu")
    config["env_gamma"] = 0.999
    config["batch_size"] = 64
    config["max_steps"] = 5*config["n_states"]
    # config["total_frames"] = 50_000
    config["total_frames"] = 100_000

    config["shortcut_steps"] = 2
    config["big_reward"] = 10
    config["return_x"] = 5
    config["return_y"] = 1

    config["agent_gamma"] = 0.999

    unconstrained_qvalues = get_qvalues(when_constraints_active=1.0, times_to_train=5, config=config)
    constrained_qvalues = get_qvalues(when_constraints_active=0.0, times_to_train=5, config=config)

    # Save dataframes in data folder
    unconstrained_qvalues.to_csv("data/unconstrained_qvalues|2025-01-27|50-states.csv", index=False)
    constrained_qvalues.to_csv("data/constrained_qvalues|2025-01-27|50-states.csv", index=False)


if __name__ == "__main__":
    main()
