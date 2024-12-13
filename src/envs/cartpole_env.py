from torchrl.envs import DMControlEnv
from torchrl.envs.transforms import (
    CatTensors,
    Compose,
    DoubleToFloat,
    RenameTransform,
    StepCounter,
    TransformedEnv,
)
import torch


def angle_normalize(x):
    return ((x + torch.pi) % (2 * torch.pi)) - torch.pi


def make_composite_from_td(td):
    # custom function to convert a ``tensordict`` in a similar spec structure
    # of unbounded values.
    composite = Composite(
        {
            key: (
                make_composite_from_td(tensor)
                if isinstance(tensor, TensorDictBase)
                else Unbounded(
                    dtype=tensor.dtype, device=tensor.device, shape=tensor.shape
                )
            )
            for key, tensor in td.items()
        },
        shape=td.shape,
    )
    return composite


def get_cartpole_env(env_config, from_pixels=False):
    def constraint_transform(td):
        # Constraint reward:
        td["constraint"] = (td["velocity"] ** 2).sum(-1, keepdim=True)
        return td

    env = TransformedEnv(
        DMControlEnv(
            "cartpole", "swingup", device=env_config["device"], from_pixels=from_pixels
        ),
        Compose(
            DoubleToFloat(),
            CatTensors(
                in_keys=["position", "velocity"], out_key="state", del_keys=False
            ),
            RenameTransform(
                in_keys=["reward"], out_keys=["normal_reward"], create_copy=True
            ),
            StepCounter(),
            constraint_transform,
        ),
    )

    return env
