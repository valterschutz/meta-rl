from tensordict import TensorDict
import os
import sys
import pickle

from torchrl.envs.utils import check_env_specs

sys.path.append(os.path.join(os.path.dirname(__file__), "../src/envs"))

from toy_env import WorkingEnv

def main():
    env = WorkingEnv()
    check_env_specs(env)

if __name__ == "__main__":
    main()
