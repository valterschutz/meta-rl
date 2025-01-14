import torch
from torchrl.envs import DMControlEnv, default_info_dict_reader
from torchrl.envs.transforms import (CatTensors, Compose, DoubleToFloat,
                                     RenameTransform, StepCounter,
                                     TransformedEnv, Transform)



class ConstraintDMControlEnv(DMControlEnv):
    def __init__(self, *args, **kwargs):
        self.constraint_weight = kwargs["constraint_weight"]
        other_kwargs = {k: v for k, v in kwargs.items() if k != "constraint_weight"}
        super().__init__(*args, **other_kwargs)
        self.set_info_dict_reader(default_info_dict_reader(["normal_reward", "constraint_reward", "weighted_reward"]))

    def _step(self, action_td):
        action = action_td["action"].clone()
        td = super()._step(action_td)
        action_magnitude = (action.abs().sum(dim=-1) / action.shape[-1]).unsqueeze(-1)
        constraint_reward = -action_magnitude  # torch.ones_like(action_magnitude) - 2 * action_magnitude
        td["normal_reward"] = td["reward"].clone()
        td["weighted_reward"] = (td["reward"] + self.constraint_weight * constraint_reward)
        td["reward"] += constraint_reward # this is used for comparing different reward functions, but not for training
        td["constraint_reward"] = constraint_reward

        # Weird torchrl thing: the "reward" key is treated in a special way
        # for key in ["normal_reward", "constraint_reward", "weighted_reward"]:
        #     td[key] = td[key].unsqueeze(-1)

        return td

    def _reset(self, tensordict = None, **kwargs):
        tensordict_out = super()._reset(tensordict, **kwargs)
        tensordict_out["normal_reward"] = torch.tensor([0.])
        tensordict_out["constraint_reward"] = torch.tensor([0.])
        tensordict_out["weighted_reward"] = torch.tensor([0.])
        return tensordict_out


def get_reacher_env(device, constraint_weight, max_steps):
    transforms = Compose(
        DoubleToFloat(),
        StepCounter(max_steps=max_steps),
        CatTensors(in_keys=["position", "velocity"], out_key="observation", del_keys=False),
        # NOTE: the "reward" key should not be used directly
    )
    env = ConstraintDMControlEnv("reacher", "easy", device=device, constraint_weight=constraint_weight)
    env = TransformedEnv(
        env,
        transforms
    )

    return env
