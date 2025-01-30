import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torchrl.data import Composite, OneHot, UnboundedContinuous, UnboundedDiscrete
from torchrl.envs import EnvBase
from torchrl.envs.transforms import (Compose, RenameTransform, StepCounter,
                                     TransformedEnv)
from torchrl.envs.utils import check_env_specs


class ToyEnv(EnvBase):

    def __init__(
        self,
        left_reward,
        right_reward,
        down_reward,
        up_reward,
        n_states,
        shortcut_steps,
        big_reward,
        punishment,
        gamma,
        constraints_active,
        random_start=False,
        seed=None,
        device="cpu",
    ):
        super().__init__(device=device, batch_size=[])

        assert (n_states-1) % shortcut_steps == 0, "n_states must be 1 more than a multiple of shortcut_steps"
        self.left_reward = left_reward
        self.right_reward = right_reward
        self.down_reward = down_reward
        self.up_reward = up_reward
        self.n_states = n_states
        self.shortcut_steps = shortcut_steps
        self.big_reward = big_reward
        self.punishment = punishment
        self.gamma = gamma
        self.random_start = random_start

        self.n_actions = 4

        self.constraints_active = constraints_active

        self._make_spec()
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    def _make_spec(self):
        self.observation_spec = Composite(
            # There does not seem to be a "BoundedDiscrete" class, so we use the unbounded one
            observation=UnboundedDiscrete(shape=(1,), dtype=torch.long),
            normal_reward=UnboundedContinuous(shape=(1,), dtype=torch.float32),
            constraint_reward=UnboundedContinuous(shape=(1,), dtype=torch.float32),
            shape=(),
        )
        self.state_spec = Composite(
            observation=UnboundedDiscrete(shape=(1,), dtype=torch.long),
            shape=(),
        )

        self.action_spec = OneHot(4, shape=(4,), dtype=torch.long)
        # The sum of normal_reward and constraint_reward
        self.reward_spec = UnboundedContinuous(shape=(1,), dtype=torch.float32)

    def _reset(self, td):
        if td is None or td.is_empty():
            batch_shape = ()
        else:
            batch_shape = td.shape

        if self.random_start:
            state_indices = torch.randint(
                0, self.n_states, shape=(*batch_shape, 1), dtype=torch.long, device=self.device
            )
        else:
            state_indices = torch.zeros((*batch_shape, 1), dtype=torch.long, device=self.device)

        state = state_indices
        # state = (
        #     F.one_hot(state_indices, num_classes=self.n_states)
        #     .to(self.device)
        # )
        # print(f"{state=}")

        out = TensorDict(
            {
                "observation": state,
                "normal_reward": torch.zeros(
                    batch_shape, dtype=torch.float32, device=self.device
                ).unsqueeze(-1),
                "constraint_reward": torch.zeros(
                    batch_shape, dtype=torch.float32, device=self.device
                ).unsqueeze(-1),
            },
            batch_size=batch_shape,
        )
        return out

    def _set_seed(self, seed):
        rng = torch.manual_seed(seed)
        self.rng = rng

    def _step(self, td):
        state = td["observation"]
        action = td["action"]  # Action order: left, right, down, up
        # state = torch.argmax(state, dim=-1)
        action = torch.argmax(action, dim=-1)

        next_state = state.clone()

        normal_reward = self.punishment * torch.ones_like(
            state, dtype=torch.float32, device=self.device
        )
        constraint_reward = 0 * torch.ones_like(
            state, dtype=torch.float32, device=self.device
        )

        left_action = 0
        right_action = 1
        down_action = 2
        up_action = 3

        # A dictionary describing where each action is possible and what rewards it gives
        d = {
            left_action: {
                "mask": action == left_action,
                "state_change": -1,
                "constraint_reward": 1 * self.left_reward,
            },
            right_action: {
                "mask": action == right_action,
                "state_change": 1,
                "constraint_reward": 1 * self.right_reward,
            },
            down_action: {
                "mask": action == down_action and (state-1) % self.shortcut_steps != 0,
                "state_change": -self.shortcut_steps if (state % self.shortcut_steps) == 0 else -(state % self.shortcut_steps),
                "constraint_reward": 1 * self.down_reward,
            },
            up_action: {
                "mask": action == up_action and (state+1) % self.shortcut_steps != 0,
                "state_change": (self.shortcut_steps-(state%self.shortcut_steps)),
                "constraint_reward": 1 * self.up_reward,
            },
        }
        action_dict = d[action.item()]

        # for action, action_dict in d.items():
        # Bound the next state between 0 and n_states-1
        next_state = torch.where(
            action_dict["mask"],
            torch.clip(next_state+action_dict["state_change"], 0, self.n_states-1),
            next_state
        )
        # constraint_reward = torch.where(
        #     action_dict["mask"], action_dict["constraint_reward"], constraint_reward
        # )
        # We always add the constraint reward, even if we don't move
        constraint_reward = torch.tensor([action_dict["constraint_reward"]], device=self.device)

        done = torch.zeros_like(state, dtype=torch.bool, device=self.device)

        # Big reward for reaching the end pos, additive with normal reward
        normal_reward = torch.where(
            next_state == self.n_states - 1, self.big_reward, normal_reward
        )
        # If we reach final pos, we're done
        # TODO: convert to bool?
        done = torch.where(next_state == self.n_states - 1, True, done)

        # For some reason, rewards have to have one dimension less
        normal_reward = normal_reward.squeeze(-1)
        constraint_reward = constraint_reward.squeeze(-1)


        reward = (
            (normal_reward + constraint_reward)
            if self.constraints_active
            else normal_reward
        )

        out = TensorDict(
            {
                "observation": next_state,
                "reward": reward,
                "normal_reward": normal_reward.unsqueeze(-1),
                "constraint_reward": constraint_reward.unsqueeze(-1),
                "done": done,
            },
            td.shape,
        )
        return out

    @staticmethod
    def calculate_xy(n_states, shortcut_steps, return_x, return_y, big_reward, gamma):
        # TODO: should work with new env
        assert (n_states-1) % shortcut_steps == 0, "n_states must be 1 more than a multiple of shortcut_steps"
        nx = n_states - 1 # Number of times we need to step 'right' to reach the end
        ny = (n_states - 1) // shortcut_steps # Number of times we need to step 'up' to reach the end
        x = (return_x - big_reward * gamma**(nx-1)) / sum(gamma**k for k in range(0, nx))
        y = (return_y - big_reward * gamma**(ny-1)) / sum(gamma**k for k in range(0, ny))
        return x, y

    def calc_optimal_qvalues(self):
        qvalues = torch.zeros(self.n_states, self.n_actions)
        delta = 1
        while delta > 1e-4:
            delta = 0
            for state in range(self.n_states-1):
                for action in range(self.n_actions):
                    td = TensorDict({
                        "observation": torch.tensor([state]),
                        "action": F.one_hot(torch.tensor(action), num_classes=self.n_actions),
                        # "step_count": torch.zeros(1),
                    })
                    td = self.step(td)
                    old_Q = qvalues[state, action].item()
                    # if td["next", "normal_reward"] != 0:
                    #     pass
                    qvalues[state, action] = td["next", "normal_reward"] + self.gamma * qvalues[td["next", "observation"], :].max()
                    delta = max(delta, abs(old_Q - qvalues[state, action].item()))
                    # print(delta)
        return qvalues

from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import tqdm
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torch import nn

from torchrl.data import Bounded, Composite, Unbounded
from torchrl.envs import (
    CatTensors,
    EnvBase,
    Transform,
    TransformedEnv,
    UnsqueezeTransform,
)
from torchrl.envs.transforms.transforms import _apply_to_composite
from torchrl.envs.utils import check_env_specs, step_mdp

def angle_normalize(x):
    return ((x + torch.pi) % (2 * torch.pi)) - torch.pi

DEFAULT_X = np.pi
DEFAULT_Y = 1.0

class WorkingEnv(EnvBase):

    batch_locked = False

    def __init__(self, td_params=None, seed=None, device="cpu"):

        if td_params is None:
            td_params = self.gen_params()

        super().__init__(device=device, batch_size=[])

        self._make_spec(td_params)
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)


    @staticmethod
    def gen_params(g=10.0, batch_size=None) -> TensorDictBase:
        """Returns a ``tensordict`` containing the physical parameters such as gravitational force and torque or speed limits."""
        if batch_size is None:
            batch_size = []
        td = TensorDict(
            {
                "params": TensorDict(
                    {
                        "max_speed": 8,
                        "max_torque": 2.0,
                        "dt": 0.05,
                        "g": g,
                        "m": 1.0,
                        "l": 1.0,
                    },
                    [],
                )
            },
            [],
        )
        if batch_size:
            td = td.expand(batch_size).contiguous()
        return td

    @staticmethod
    def _step(tensordict):
        th, thdot = tensordict["th"], tensordict["thdot"]  # th := theta

        reward = torch.zeros_like(th, dtype=torch.float32).unsqueeze(-1)
        done = torch.zeros_like(th, dtype=torch.bool)
        out = TensorDict(
            {
                "th": th,
                "thdot": thdot,
                # "params": tensordict["params"],
                "reward": reward,
                "done": done,
            },
            tensordict.shape,
        )
        return out

    def _reset(self, tensordict):
        if tensordict is None or tensordict.is_empty():
            # if no ``tensordict`` is passed, we generate a single set of hyperparameters
            # Otherwise, we assume that the input ``tensordict`` contains all the relevant
            # parameters to get started.
            tensordict = self.gen_params(batch_size=self.batch_size)

        high_th = torch.tensor(DEFAULT_X, device=self.device)
        high_thdot = torch.tensor(DEFAULT_Y, device=self.device)
        low_th = -high_th
        low_thdot = -high_thdot

        # for non batch-locked environments, the input ``tensordict`` shape dictates the number
        # of simulators run simultaneously. In other contexts, the initial
        # random state's shape will depend upon the environment batch-size instead.
        th = (
            torch.rand(tensordict.shape, generator=self.rng, device=self.device)
            * (high_th - low_th)
            + low_th
        )
        thdot = (
            torch.rand(tensordict.shape, generator=self.rng, device=self.device)
            * (high_thdot - low_thdot)
            + low_thdot
        )
        out = TensorDict(
            {
                "th": th,
                "thdot": thdot,
            },
            batch_size=tensordict.shape,
        )
        return out

    def _make_spec(self, td_params):
        # Under the hood, this will populate self.output_spec["observation"]
        self.observation_spec = Composite(
            th=Bounded(
                low=-torch.pi,
                high=torch.pi,
                shape=(),
                dtype=torch.float32,
            ),
            thdot=Bounded(
                low=-td_params["params", "max_speed"],
                high=td_params["params", "max_speed"],
                shape=(),
                dtype=torch.float32,
            ),
            # we need to add the ``params`` to the observation specs, as we want
            # to pass it at each step during a rollout
            # params=make_composite_from_td(td_params["params"]),
            shape=(),
        )
        # since the environment is stateless, we expect the previous output as input.
        # For this, ``EnvBase`` expects some state_spec to be available
        self.state_spec = self.observation_spec.clone()
        # action-spec will be automatically wrapped in input_spec when
        # `self.action_spec = spec` will be called supported
        self.action_spec = Bounded(
            low=-td_params["params", "max_torque"],
            high=td_params["params", "max_torque"],
            shape=(1,),
            dtype=torch.float32,
        )
        self.reward_spec = Unbounded(shape=(*td_params.shape, 1))

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng


def make_composite_from_td(td):
    # custom function to convert a ``tensordict`` in a similar spec structure
    # of unbounded values.
    composite = Composite(
        {
            key: make_composite_from_td(tensor)
            if isinstance(tensor, TensorDictBase)
            else Unbounded(dtype=tensor.dtype, device=tensor.device, shape=tensor.shape)
            for key, tensor in td.items()
        },
        shape=td.shape,
    )
    return composite
