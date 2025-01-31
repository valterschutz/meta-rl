from tensordict import TensorDict
import torch
import torch.nn.functional as F
import os
import sys
import pickle

from torchrl.envs.utils import check_env_specs
from torchrl.collectors import SyncDataCollector

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from envs.toy_env import WorkingEnv, ToyEnv
from agents.base_agents import fast_policy, slow_policy, ToyTabularQAgent

def main():
    env = ToyEnv(
        left_reward=-1,
        right_reward=-1,
        down_reward=-3,
        up_reward=-3,
        n_states=16,
        shortcut_steps=3,
        big_reward=10.0,
        punishment=0.0,
        gamma=0.99,
        constraints_active=False,
        random_start=False,
        seed=None,
        device="cpu"
    )
    check_env_specs(env)

    agent = ToyTabularQAgent(
        n_states=16,
        gamma=0.99,
        lr=1e-2,
        epsilon=0.1,
        replay_buffer_size=100,
        device="cpu",
        rb_alpha=0.7,
        rb_beta=0.5,
        rb_batch_size=10,
        num_optim_steps=1
    )
    collector = SyncDataCollector(
        env,
        # None,
        agent.train_policy,
        frames_per_batch=64,
        total_frames=640,
        split_trajs=False,
        device="cpu",
    )
    # collector = SyncDataCollector(
    #     env,
    #     agent.train_policy,
    #     frames_per_batch=64,
    #     total_frames=640,
    #     split_trajs=False,
    #     device="cpu",
    # )

    for td in collector:
        pass


    print("Success")

if __name__ == "__main__":
    main()
