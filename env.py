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
        self,
        x,
        y,
        n_pos,
        big_reward,
        random_start=False,
        punishment=False,
        seed=None,
        device="cpu",
    ):
        super().__init__(device=device, batch_size=[])

        assert n_pos % 2 == 0, "n_pos only tested for even numbers"
        self.x = x
        self.y = y
        self.n_pos = n_pos
        self.big_reward = big_reward
        self.random_start = random_start
        self.punishment = punishment

        # self.params = self.gen_params(x, y, n_pos, big_reward)

        self._make_spec()
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    def _make_spec(self):
        self.observation_spec = Composite(
            pos=Categorical(self.n_pos, shape=(), dtype=torch.int32),
            shape=(),
        )
        self.state_spec = Composite(
            pos=Categorical(self.n_pos, shape=(), dtype=torch.int32), shape=()
        )
        self.action_spec = Categorical(4, shape=(), dtype=torch.int32)
        self.reward_spec = Unbounded(
            shape=(1,),  # WHY
            dtype=torch.int32,
            domain="discrete",
        )

    def _reset(self, td):
        if td is None or td.is_empty():
            shape = ()
        else:
            shape = td.shape

        if self.random_start:
            pos = torch.randint(
                0, self.n_pos, shape=shape, dtype=torch.int32, device=self.device
            )
        else:
            pos = torch.zeros(shape, dtype=torch.int32, device=self.device)

        out = TensorDict(
            {
                "pos": pos,
            },
            batch_size=shape,
        )
        return out

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng

    def _step(self, td):
        pos = td["pos"]
        action = td["action"]  # Action order: left, right, down, up
        x, y, n_pos, big_reward = self.x, self.y, self.n_pos, self.big_reward

        next_pos = pos.clone()
        reward = 0 * torch.ones_like(pos, dtype=torch.int)

        mask_even = pos % 2 == 0

        # Enable left action by default
        next_pos = torch.where(action == 0, pos - 1, next_pos)
        reward = torch.where(action == 0, -x, reward)
        # Enable right action by default
        next_pos = torch.where(action == 1, pos + 1, next_pos)
        reward = torch.where(action == 1, -x, reward)

        # For even pos, enable down and up actions
        # Down action
        next_pos = torch.where(mask_even & (action == 2), pos - 2, next_pos)
        reward = torch.where(mask_even & (action == 2), -y, reward)
        # Up action
        next_pos = torch.where(mask_even & (action == 3), pos + 2, next_pos)
        reward = torch.where(mask_even & (action == 3), -y, reward)

        # Ensure that we can never move past the end pos
        next_pos = torch.where(next_pos >= n_pos, n_pos - 1, next_pos)

        # Ensure that we can never move before the start pos
        next_pos = torch.where(next_pos < 0, pos, next_pos)

        # If we did not move, terminate the episode and (maybe) punish
        done = torch.where(next_pos == pos, 1.0, 0.0).to(torch.bool)
        if self.punishment:
            reward = torch.where(next_pos == pos, -big_reward, reward)
        else:
            reward = torch.where(next_pos == pos, 0, reward)

        # Big reward for reaching the end pos
        reward = torch.where(next_pos == n_pos - 1, big_reward, reward)
        # If we reach final pos, we're done
        done = torch.where(next_pos == n_pos - 1, 1.0, done).to(torch.bool)

        out = TensorDict(
            {
                "pos": next_pos,
                "reward": reward,
                "done": done,
            },
            td.shape,
        )
        return out
