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

    def __init__(
        self, x, y, n_pos, big_reward, random_start=False, seed=None, device="cpu"
    ):
        super().__init__(device=device, batch_size=[])

        assert n_pos % 2 == 0, "n_pos only tested for even numbers"
        self.x = x
        self.y = y
        self.n_pos = n_pos
        self.big_reward = big_reward
        self.random_start = random_start

        self.params = self.gen_params(x, y, n_pos, big_reward)

        self._make_spec(self.params)
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
            # shape=(1),  # WHY
            dtype=torch.int,
            domain="discrete",
        )

    @staticmethod
    def gen_params(x, y, n_pos, big_reward, batch_size=None) -> TensorDictBase:
        """Returns a ``tensordict`` containing the physical parameters such as gravitational force and torque or speed limits."""
        if batch_size is None:
            batch_size = []
        td = TensorDict(
            {
                "params": TensorDict(
                    {"x": x, "y": y, "n_pos": n_pos, "big_reward": big_reward},
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
            td = self.gen_params(
                self.x, self.y, self.n_pos, self.big_reward, batch_size=self.batch_size
            )

        if self.random_start:
            pos = torch.randint(0, td["params", "n_pos"], td.shape, device=self.device)
        else:
            pos = torch.zeros(td.shape, device=self.device, dtype=torch.long)
        # Convert into OneHot
        # pos = torch.nn.functional.one_hot(pos, td["params", "n_pos"])
        # print(f"at reset: {pos.shape=}")

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
        action = td["action"]  # Action order: left, right, down, up
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

        next_pos = pos.clone()
        reward = torch.zeros_like(pos, dtype=torch.int)

        mask_start = pos == 0
        mask_end = pos == (n_pos - 1)
        mask_even = pos % 2 == 0
        mask_before_end = pos == (n_pos - 2)  # Right before the end pos

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

        # Ensure that we can never move past the end pos
        next_pos = torch.where(next_pos >= n_pos, n_pos - 1, next_pos)

        # Ensure that we can never move before the start pos
        next_pos = torch.where(next_pos < 0, pos, next_pos)

        # For starting pos, disable left action
        # next_pos = torch.where(mask_start & (action == 0), pos, next_pos)
        # reward = torch.where(mask_start & (action == 0), 0, reward)
        # For starting pos, disable down action
        # next_pos = torch.where(mask_start & (action == 2), pos, next_pos)
        # reward = torch.where(mask_start & (action == 2), 0, reward)

        # For end pos, disable right action
        # next_pos = torch.where(mask_end & (action == 1), pos, next_pos)
        # reward = torch.where(mask_end & (action == 1), 0, reward)

        # If we reach final pos, we're done
        done = torch.where(next_pos == n_pos - 1, 1.0, 0.0).to(torch.bool)

        # Big reward for reaching the end pos
        reward = torch.where(next_pos == n_pos - 1, big_reward, reward).to(torch.int)

        # No rewards in terminal state
        # reward = torch.where(done, 0, reward).to(torch.int)

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
