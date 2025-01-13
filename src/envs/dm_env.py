import torch
from torchrl.envs import DMControlEnv
from torchrl.envs.transforms import (CatTensors, Compose, DoubleToFloat,
                                     RenameTransform, StepCounter,
                                     TransformedEnv)

def get_cartpole_env(env_config):
    def constraint_transform(td):
        # Constraint reward:
        td["constraint_reward"] = (td["velocity"] ** 2).sum(-1, keepdim=True)
        if "next" in td:
            td["next","constraint_reward"] = (td["next", "velocity"] ** 2).sum(-1, keepdim=True)

        return td

    def f(from_pixels):
        return TransformedEnv(
            DMControlEnv(
                "cartpole",
                "swingup",
                device=env_config["device"],
                from_pixels=from_pixels,
            ),
            Compose(
                DoubleToFloat(),
                CatTensors(
                    in_keys=["position", "velocity"], out_key="state", del_keys=False
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

def get_reacher_env(env_config):
    def constraint_transform(td):
        # Constraint reward:
        td["constraint_reward"] = (td["velocity"] ** 2).sum(-1, keepdim=True)
        if "next" in td:
            td["next","constraint_reward"] = (td["next", "velocity"] ** 2).sum(-1, keepdim=True)

        return td

    def f(from_pixels):
        return TransformedEnv(
            DMControlEnv(
                "reacher",
                "easy",
                device=env_config["device"],
                from_pixels=from_pixels,
            ),
            Compose(
                DoubleToFloat(),
                CatTensors(
                    in_keys=["position", "velocity"], out_key="state", del_keys=False
                ),
                RenameTransform(
                    in_keys=["reward"], out_keys=["normal_reward"], create_copy=True
                ),
                RenameTransform(
                    in_keys=["reward"], out_keys=["constraint_reward"], create_copy=True
                ),
                # RenameTransform(
                #     in_keys=["velocity"], out_keys=["constraint_reward"], create_copy=True
                # ),
                StepCounter(max_steps=env_config["max_steps"]),
                # constraint_transform, # TODO: make constraints work
            ),
        )

    env = f(from_pixels=False)
    pixel_env = f(from_pixels=True)
    return env, pixel_env

def get_point_mass_env(env_config):
    def constraint_transform(td):
        # Constraint reward:
        td["constraint_reward"] = (td["velocity"] ** 2).sum(-1, keepdim=True)
        if "next" in td:
            td["next","constraint_reward"] = (td["next", "velocity"] ** 2).sum(-1, keepdim=True)

        return td

    def f(from_pixels):
        return TransformedEnv(
            DMControlEnv(
                "point_mass",
                "easy",
                device=env_config["device"],
                from_pixels=from_pixels,
            ),
            Compose(
                DoubleToFloat(),
                CatTensors(
                    in_keys=["position", "velocity"], out_key="state", del_keys=False
                ),
                RenameTransform(
                    in_keys=["reward"], out_keys=["normal_reward"], create_copy=True
                ),
                RenameTransform(
                    in_keys=["reward"], out_keys=["constraint_reward"], create_copy=True
                ),
                # RenameTransform(
                #     in_keys=["velocity"], out_keys=["constraint_reward"], create_copy=True
                # ),
                StepCounter(max_steps=env_config["max_steps"]),
                # constraint_transform, # TODO: make constraints work
            ),
        )

    env = f(from_pixels=False)
    pixel_env = f(from_pixels=True)
    return env, pixel_env

class ConstraintDMControlEnv(DMControlEnv):
    def __init__(self, *args, **kwargs):
        self.constraint_weight = kwargs["constraint_weight"]
        other_kwargs = {k: v for k, v in kwargs.items() if k != "constraint_weight"}
        super().__init__(*args, **other_kwargs)

    def _step(self, action_td):
        td = super()._step(action_td)
        # scale action to be between -1 and 1
        # norm_action = 2 * (action_td["action"] - self.action_spec.low) / (self.action_spec.high - self.action_spec.low) - 1
        # calculate reward in [0, 1]
        action = action_td["action"]
        constraint_reward = (action.abs().sum(dim=-1) / action.shape[-1]).unsqueeze(-1)
        td["normal_reward"] = td["reward"].clone()
        td["reward"] += self.constraint_weight * constraint_reward
        td["constraint_reward"] = constraint_reward
        return td

    # def _reset(self, td):
    #     td = super()._reset(td)
    #     td["normal_reward"] = td["reward"].clone()
    #     return td
