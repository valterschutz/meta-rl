import os
import sys
import yaml
import argparse

from importlib import import_module

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from trainers import OffpolicyTrainer
from algorithms import get_sac_alg, get_dqn_alg
from agents import Agent

def get_obj(path_fun_str):
    """
    Returns the object that is returned by the function at the specified path.
    The path should be in the form '<path>.<fun_name>'.
    """
    module = import_module(path_fun_str.split('.')[0])
    fun = getattr(module, path_fun_str.split('.')[1])
    return fun()

def get_config(path):
    """
    Returns a configuration object from the specified yaml file.
    """
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def main():

    argparser = argparse.ArgumentParser()
    env_group = argparser.add_argument_group("Environment")
    env_group.add_argument("--env", type=str, required=True, help="Path to a function that returns an environment object, in the form '<path>.<fun_name>'.")
    env_group.add_argument("--env-config", type=str, required=True, help="Path to environment yaml configuration file.")

    # The algorithm is just a container for a torchrl.objectives loss function
    agent_group = argparser.add_argument_group("Algorithm")
    agent_group.add_argument("--rl-alg", type=str, required=True, help="String specififying the RL algorithm (e.g., SAC, DQN).")
    agent_group.add_argument("--policy-network", type=str, help="Path to a a function that returns a 'policy_network' object, in the form '<path>.<fun_name>'.")
    agent_group.add_argument("--state-value-network", type=str, help="Path to a function that returns a 'state_value_network' object, in the form '<path>.<fun_name>'.")
    agent_group.add_argument("--action-value-network", type=str, help="Path to a function that returns an 'action_value_network' object, in the form '<path>.<fun_name>'.")
    agent_group.add_argument("--alg-config", type=str, help="Path to algorithm yaml configuration file.")

    # An agent contains an algorithm and replay buffer. It has a method for processing a batch of data.
    agent_group = argparser.add_argument_group("Agent")
    agent_group.add_argument("--agent-config", type=str, help="Path to agent yaml configuration file.")

    # The trainer contains an agent and an environment. It constructs a collector and uses its data to train the agent.
    agent_group = argparser.add_argument_group("Trainer")
    agent_group.add_argument("--trainer-config", type=str, help="Path to trainer yaml configuration file.")

    args = argparser.parse_args()

    env = get_obj(args.env)
    if args.rl_alg == "SAC":
        alg = get_sac_alg(
            policy_network=get_obj(args.policy_network),
            action_value_network=get_obj(args.action_value_network),
            alg_config=get_config(args.alg_config)
        )
    elif args.rl_alg == "DQN":
        alg = get_dqn_alg(
            policy_network=get_obj(args.policy_network),
            action_value_network=get_obj(args.action_value_network),
            alg_config=get_obj(args.alg_config)
        )
    agent_config = get_config(args.agent_config)
    agent = Agent(alg, agent_config)

    trainer_config = get_config(args.trainer_config)
    trainer = OffpolicyTrainer(env, agent, trainer_config)

    trainer.train()







wandb.init(project="toy-base")

device = torch.device("cpu")
batch_size = 64
gamma = 0.999
n_states = 500
max_steps = 5*n_states
shortcut_steps = 2
big_reward = 10
return_x = 5
return_y = 1
total_frames = 100_000

x, y = ToyEnv.calculate_xy(n_states=n_states, shortcut_steps=shortcut_steps, return_x=return_x, return_y=return_y, big_reward=big_reward, punishment=0, gamma=gamma)

env = ToyEnv(
    left_reward=x,
    right_reward=x,
    down_reward=y,
    up_reward=y,
    n_states=n_states,
    shortcut_steps=shortcut_steps,
    big_reward=big_reward,
    punishment=0.0,
    constraints_active=False,
    random_start=False,
    seed=None,
    device=device)
env = TransformedEnv(
    env,
    Compose(
        StepCounter(max_steps=max_steps),
        DTypeCastTransform(dtype_in=torch.long, dtype_out=torch.float32, in_keys=["observation"]),
    )
)

agent = ToyDQNAgent(
    device=device,
    batch_size=batch_size,
    sub_batch_size=batch_size,
    num_epochs=1,
    replay_buffer_args={
        "buffer_size": total_frames,
        "min_buffer_size": 0,
        "alpha": 0.7,
        "beta": 0.5
    },
    env=env,
    agent_detail_args={
        "agent_gamma": gamma,
        "target_eps": 0.99,
        "value_lr": 1e-3,
        "value_max_grad": 10,
        "num_cells": [32, 32],
        "qvalue_eps": 0.1
    }
)


trainer = OffpolicyTrainer(
    env=env,
    agent=agent,
    progress_bar=True,
    times_to_eval=20,
    collector_device=device,
    log=True,
    max_eval_steps=max_steps,
    collector_args={
        "batch_size": batch_size,
        "total_frames": total_frames,
    },
    env_gamma=gamma,
    eval_env=None
)
trainer.train(when_constraints_active=0.99)
