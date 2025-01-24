import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/envs"))

from toy_env import ToyEnv

def get_toy_env():
    return ToyEnv(
        left_reward=1,
        right_reward=1,
        down_reward=1,
        up_reward=1,
        n_states=500,
        shortcut_steps=2,
        big_reward=10,
        punishment=0.0,
        constraints_active=False,
        random_start=False,
        seed=None,
        device="cpu",
    )
