from tensordict import TensorDict
import torch
import torch.nn.functional as F
import os
import sys
import pickle

from torchrl.envs.utils import check_env_specs

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from envs.toy_env import WorkingEnv, ToyEnv
from agents.base_agents import fast_policy, slow_policy

def main():
    env = ToyEnv(
        # batch_size=(16*4,),
        batch_size=(1,),
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

    td = env.reset()
    # td["observation"] = torch.arange(16).repeat(4).unsqueeze(-1)
    td["observation"] = torch.tensor([1]).unsqueeze(-1)
    td = fast_policy(td)
    td = env.step(td)

    td = env.rollout(1000, slow_policy)

    print("Success")

if __name__ == "__main__":
    main()
