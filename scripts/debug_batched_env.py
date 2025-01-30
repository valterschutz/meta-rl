from tensordict import TensorDict
import torch
import os
import sys
import pickle

from torchrl.envs.utils import check_env_specs

sys.path.append(os.path.join(os.path.dirname(__file__), "../src/envs"))

from toy_env import WorkingEnv

def main():
    env = WorkingEnv(batch_size=(7,))
    check_env_specs(env)

    # Create a batch of data
    # td = TensorDict({
    #     "th": torch.tensor([1.0, 2.0, 3.0]),
    #     "thdot": torch.tensor([0.0, 0.0, 0.0]),
    # })
    td = env.reset()
    td["th"] = torch.ones_like(td["th"]) * 0.5
    td["thdot"] = torch.ones_like(td["thdot"])
    td = env.step(td)

if __name__ == "__main__":
    main()
