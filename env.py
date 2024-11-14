from typing import Optional

import torch
import tqdm

from tensordict import TensorDict, TensorDictBase

from torchrl.data import Bounded, Composite, Unbounded, Categorical
from torchrl.envs import (
    EnvBase,
)


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


class ToyEnv(EnvBase):
    batch_locked = False

    def __init__(self, n_pos, seed=None, device="cpu"):
        super().__init__(device=device, batch_size=[])

        self.n_pos = n_pos

        td_params = self.gen_params(n_pos)

        self._make_spec(td_params)
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    def _make_spec(self, td_params):
        self.observation_spec = Composite(
            # state=Bounded(low=0, high=, shape=(), dtype=torch.int, domain="discrete"),
            pos=Categorical(td_params["params", "n_pos"]),
            params=make_composite_from_td(td_params["params"]),
            shape=(),
        )
        # self.state_spec = self.observation_spec.clone()
        self.state_spec = Composite(
            pos=Categorical(td_params["params", "n_pos"]), shape=()
        )
        # self.action_spec = Bounded(
        #     low=0,
        #     high=3,
        #     shape=(),
        #     dtype=torch.int,
        # )
        self.action_spec = Categorical(4)
        self.reward_spec = Unbounded(
            shape=(*td_params.shape, 1),  # WHY
            dtype=torch.int,
            domain="discrete",
        )

    @staticmethod
    def gen_params(n_pos, batch_size=None) -> TensorDictBase:
        """Returns a ``tensordict`` containing the physical parameters such as gravitational force and torque or speed limits."""
        if batch_size is None:
            batch_size = []
        td = TensorDict(
            {
                "params": TensorDict(
                    {"x": 1, "y": 3, "n_pos": n_pos, "big_reward": 100},
                    [],
                )
            },
            [],
        )
        if batch_size:
            td = td.expand(batch_size).contiguous()
        return td

    _make_spec = _make_spec

    def _reset(self, td):
        if td is None or td.is_empty():
            td = self.gen_params(self.n_pos, batch_size=self.batch_size)

        pos = torch.zeros(td.shape, device=self.device, dtype=torch.long)
        # Convert into OneHot
        # pos = torch.nn.functional.one_hot(pos, td["params", "n_pos"])
        print(f"at reset: {pos.shape=}")

        out = TensorDict(
            {
                "pos": pos,
                "params": td["params"],
            },
            batch_size=td.shape,
        )
        return out

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng

    @staticmethod
    def _step(td):
        pos = td["pos"]
        # print(f"before argmax: {pos.shape=}")
        action = td["action"]
        # print(f"action: {action.shape=}")
        x, y, n_pos, big_reward = (
            td["params", "x"],
            td["params", "y"],
            td["params", "n_pos"],
            td["params", "big_reward"],
        )

        # Convert pos and action from OneHot to integers
        # pos = torch.argmax(pos.to(torch.long), dim=-1)
        # print(f"after argmax: {pos.shape=}")
        # action = torch.argmax(action.to(torch.long), dim=-1)

        # If the pos is 1 and the action is 1, the reward is -x
        next_pos = (
            pos.clone()
        )  # If no next pos is defined, the next pos is the same as the current pos
        reward = torch.zeros_like(
            pos, dtype=torch.int
        )  # If no reward is defined, the reward is 0

        mask_start = pos == 0
        mask_end = pos == n_pos
        mask_even = pos % 2 == 0

        # Enable left action by default
        next_pos = torch.where(action == 0, pos - 1, next_pos)
        reward = torch.where(action == 0, -x.to(torch.int), reward)
        # Enable right action by default
        next_pos = torch.where(action == 1, pos + 1, next_pos)
        reward = torch.where(action == 1, -x.to(torch.int), reward)

        # For even poss, enable down and up actions
        # Down action
        next_pos = torch.where(mask_even & (action == 2), pos - 2, next_pos)
        reward = torch.where(mask_even & (action == 2), -y.to(torch.int), reward)
        # Up action
        next_pos = torch.where(mask_even & (action == 3), pos + 2, next_pos)
        reward = torch.where(mask_even & (action == 3), -y.to(torch.int), reward)

        # For starting pos, disable left action
        next_pos = torch.where(mask_start & (action == 0), pos, next_pos)
        reward = torch.where(mask_start & (action == 0), 0, reward)
        # For starting pos, disable down action
        next_pos = torch.where(mask_start & (action == 2), pos, next_pos)
        reward = torch.where(mask_start & (action == 2), 0, reward)

        # For end pos, disable right action
        next_pos = torch.where(mask_end & (action == 1), pos, next_pos)
        reward = torch.where(mask_end & (action == 1), 0, reward)
        # End pos is done
        done = torch.where(mask_end, 1.0, 0.0).to(torch.bool)

        # Big reward for reaching the end pos
        reward = torch.where(mask_end, big_reward, reward).to(torch.int)

        # print(f"before one_hot: {next_pos.shape=}")
        # Convert next_pos into OneHot
        # next_pos = torch.nn.functional.one_hot(next_pos, n_pos)
        # print(f"after one_hot: {next_pos.shape=}")

        out = TensorDict(
            {
                "pos": next_pos,
                "params": td["params"],
                "reward": reward,
                "done": done,
            },
            td.shape,
        )
        return out
