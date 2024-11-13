from typing import Optional

import torch
import tqdm

from tensordict import TensorDict, TensorDictBase

from torchrl.data import Bounded, Composite, Unbounded
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

    def __init__(self, td_params=None, seed=None, device="cpu"):
        if td_params is None:
            td_params = self.gen_params()

        super().__init__(device=device, batch_size=[])
        self._make_spec(td_params)
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    def _make_spec(self, td_params):
        self.observation_spec = Composite(
            state=Bounded(low=0, high=3, shape=(), dtype=torch.int, domain="discrete"),
            params=make_composite_from_td(td_params["params"]),
            shape=(),
        )
        self.state_spec = self.observation_spec.clone()
        self.action_spec = Bounded(
            low=0,
            high=3,
            shape=(),
            dtype=torch.int,
        )
        self.reward_spec = Unbounded(
            shape=(*td_params.shape, 1),  # WHY
            dtype=torch.int,
            domain="discrete",
        )

    @staticmethod
    def gen_params(batch_size=None) -> TensorDictBase:
        """Returns a ``tensordict`` containing the physical parameters such as gravitational force and torque or speed limits."""
        if batch_size is None:
            batch_size = []
        td = TensorDict(
            {
                "params": TensorDict(
                    {"x": 1, "y": 3},
                    [],
                )
            },
            [],
        )
        if batch_size:
            td = td.expand(batch_size).contiguous()
        return td

    _make_spec = _make_spec

    def _reset(self, tensordict):
        if tensordict is None or tensordict.is_empty():
            tensordict = self.gen_params(batch_size=self.batch_size)

        state = torch.zeros(tensordict.shape, device=self.device, dtype=torch.int)

        out = TensorDict(
            {
                "state": state,
                "params": tensordict["params"],
            },
            batch_size=tensordict.shape,
        )
        return out

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng

    @staticmethod
    def _step(tensordict):
        state = tensordict["state"]
        action = tensordict["action"]
        x, y = tensordict["params", "x"], tensordict["params", "y"]

        # If the state is 1 and the action is 1, the reward is -x
        next_state = (
            state.clone()
        )  # If no next state is defined, the next state is the same as the current state
        reward = torch.zeros_like(
            state, dtype=torch.int
        )  # If no reward is defined, the reward is 0

        # TODO: define these somewhere else
        n_states = 5
        big_reward = 100

        mask_start = state == 0
        mask_end = state == n_states
        mask_even = state % 2 == 0

        # Enable left action by default
        next_state = torch.where(action == 0, state - 1, next_state)
        reward = torch.where(action == 0, -x.to(torch.int), reward)
        # Enable right action by default
        next_state = torch.where(action == 1, state + 1, next_state)
        reward = torch.where(action == 1, -x.to(torch.int), reward)

        # For even states, enable down and up actions
        # Down action
        next_state = torch.where(mask_even & (action == 2), state - 2, next_state)
        reward = torch.where(mask_even & (action == 2), -y.to(torch.int), reward)
        # Up action
        next_state = torch.where(mask_even & (action == 3), state + 2, next_state)
        reward = torch.where(mask_even & (action == 3), -y.to(torch.int), reward)

        # For starting state, disable left action
        next_state = torch.where(mask_start & (action == 0), state, next_state)
        reward = torch.where(mask_start & (action == 0), 0, reward)
        # For starting state, disable down action
        next_state = torch.where(mask_start & (action == 2), state, next_state)
        reward = torch.where(mask_start & (action == 2), 0, reward)

        # For end state, disable right action
        next_state = torch.where(mask_end & (action == 1), state, next_state)
        reward = torch.where(mask_end & (action == 1), 0, reward)
        # End state is done
        done = torch.where(mask_end, 1.0, 0.0).to(torch.bool)

        # Big reward for reaching the end state
        reward = torch.where(mask_end, big_reward, reward)

        out = TensorDict(
            {
                "state": next_state,
                "params": tensordict["params"],
                "reward": reward,
                "done": done,
            },
            tensordict.shape,
        )
        return out
