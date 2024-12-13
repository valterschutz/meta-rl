import os
import sys
from pathlib import Path

import torch
import yaml
from torchrl.envs.transforms import TransformedEnv
from torchrl.record.loggers.csv import CSVLogger
from torchrl.record import VideoRecorder

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.envs.cartpole_env import get_cartpole_env


def main():
    with open("configs/envs/cartpole_env.yaml", encoding="UTF-8") as f:
        env_config = yaml.safe_load(f)

    env = get_cartpole_env(env_config, from_pixels=True)
    # logger = CSVLogger(
    #     exp_name="cartpole", log_dir="logs/", video_format="mp4", video_fps=30
    # )
    logger = CSVLogger(exp_name="cartpole", log_dir="logs", video_format="mp4")
    recorder = VideoRecorder(logger, tag="my_video", fps=30)
    record_env = TransformedEnv(env, recorder)

    # Load model weights
    print("Loading policy...")
    algo = sys.argv[1]
    p = Path(f"models/cartpole/{algo}/")
    # p has several directories inside, each in a datetime format. We will load the latest one.
    p = max(p.iterdir(), key=os.path.getctime)
    # Create the directory if not already present
    policy_path = p / "policy.pth"
    policy = torch.load(policy_path)

    # Rollout with policy
    print("Rollout...")
    rollout = record_env.rollout(max_steps=100)
    # recorder.dump()
    record_env.transform.dump()


if __name__ == "__main__":
    main()
