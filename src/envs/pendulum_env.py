import torch
from torchrl.envs import DMControlEnv
from torchrl.envs.transforms import (CatTensors, Compose, DoubleToFloat,
                                     RenameTransform, StepCounter,
                                     TransformedEnv)

def get_pendulum_env(env_config):
    def constraint_transform(td):
        # Constraint reward:
        td["constraint_reward"] = (td["velocity"] ** 2).sum(-1, keepdim=True)
        if "next" in td:
            td["next","constraint_reward"] = (td["next", "velocity"] ** 2).sum(-1, keepdim=True)

        return td

    def f(from_pixels):
        return TransformedEnv(
            DMControlEnv(
                "pendulum",
                "swingup",
                device=env_config["device"],
                from_pixels=from_pixels,
            ),
            Compose(
                DoubleToFloat(),
                CatTensors(
                    in_keys=["orientation", "velocity"], out_key="state", del_keys=False
                ),
                RenameTransform(
                    in_keys=["reward"], out_keys=["normal_reward"], create_copy=True
                ),
                RenameTransform(
                    in_keys=["velocity"], out_keys=["constraint_reward"], create_copy=True
                ),
                StepCounter(max_steps=env_config["max_steps"]),
                # constraint_transform, # TODO: make constraints work
            ),
        )

    env = f(from_pixels=False)
    pixel_env = f(from_pixels=True)
    return env, pixel_env
