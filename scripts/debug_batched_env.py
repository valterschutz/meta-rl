from tensordict import TensorDict
import torch
import torch.nn.functional as F
import os
import sys
import pickle

from torchrl.envs.utils import check_env_specs

sys.path.append(os.path.join(os.path.dirname(__file__), "../src/envs"))

from toy_env import WorkingEnv, ToyEnv

# def main():
#     env = WorkingEnv(batch_size=(7,))
#     check_env_specs(env)

#     # Create a batch of data
#     td = env.reset()
#     td["th"] = torch.ones_like(td["th"]) * 0.5
#     td["thdot"] = torch.ones_like(td["thdot"])
#     td = env.step(td)

def main():
    env = ToyEnv(
        batch_size=(16*4,),
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
    # check_env_specs(env)

    # Create a batch of data
    # td = TensorDict({
    #     "th": torch.tensor([1.0, 2.0, 3.0]),
    #     "thdot": torch.tensor([0.0, 0.0, 0.0]),
    # })
    td = env.reset()
    td["observation"] = torch.arange(16).repeat(4).unsqueeze(-1)
    td["action"] = F.one_hot(torch.arange(4).repeat_interleave(16), num_classes=4)
    td = env.step(td)
    print("hello")

if __name__ == "__main__":
    main()
