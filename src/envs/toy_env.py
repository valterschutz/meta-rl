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
        batch_size,
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
        super().__init__(device=device, batch_size=batch_size)

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
            observation=UnboundedDiscrete(shape=(*self.batch_size, 1), dtype=torch.long),
            normal_reward=UnboundedContinuous(shape=(*self.batch_size, 1), dtype=torch.float32),
            constraint_reward=UnboundedContinuous(shape=(*self.batch_size,1), dtype=torch.float32),
            shape=self.batch_size,
        )
        self.state_spec = Composite(
            observation=UnboundedDiscrete(shape=(*self.batch_size,1), dtype=torch.long),
            shape=self.batch_size,
        )

        self.action_spec = OneHot(4, shape=(*self.batch_size,4), dtype=torch.long)
        # The sum of normal_reward and constraint_reward
        self.reward_spec = UnboundedContinuous(shape=(*self.batch_size,1), dtype=torch.float32)

    def _reset(self, td):
        if td is None or td.is_empty():
            batch_size = self.batch_size
        else:
            batch_size = td.shape

        if self.random_start:
            state_indices = torch.randint(
                0, self.n_states, shape=(*batch_size, 1), dtype=torch.long, device=self.device
            )
        else:
            state_indices = torch.zeros((*batch_size, 1), dtype=torch.long, device=self.device)

        state = state_indices

        out = TensorDict(
            {
                "observation": state,
                "normal_reward": torch.zeros(
                    (*batch_size,1), dtype=torch.float32, device=self.device
                ),
                "constraint_reward": torch.zeros(
                    (*batch_size, 1), dtype=torch.float32, device=self.device
                )
            },
            batch_size=batch_size,
        )
        return out

    def _set_seed(self, seed):
        rng = torch.manual_seed(seed)
        self.rng = rng

    def _step(self, td):
        state = td["observation"].squeeze(-1) # Squeeze last dimension to simplify tensor calculations
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

        # Create masks for each action
        masks = {
            "left": action == left_action,
            "right": action == right_action,
            "down": action == down_action,
            "up": action == up_action,
        }

        # Compute state changes and rewards for each action
        state_changes = {
            "left": -1,
            "right": 1,
            # Trying to move down one state after a checkpoint will not move us
            # Otherwise move to the previous checkpoint
            "down": torch.where(
                ((state-1) % self.shortcut_steps) == 0,
                0,
                torch.where(
                    (state % self.shortcut_steps) == 0,
                    -self.shortcut_steps,
                    -(state % self.shortcut_steps),
                )
            ),
            # Trying to move up one state before a checkpoint will not move us
            # Otherwise move to the next checkpoint
            "up": torch.where(
                ((state+1) % self.shortcut_steps) == 0,
                0,
                self.shortcut_steps - (state % self.shortcut_steps),
            )
        }

        constraint_rewards = {
            "left": self.left_reward,
            "right": self.right_reward,
            "down": self.down_reward,
            "up": self.up_reward,
        }

        # Apply changes based on action masks
        for action_name in ["left", "right", "down", "up"]:
            mask = masks[action_name]
            next_state = torch.where(
                mask,
                torch.clip(next_state + state_changes[action_name], 0, self.n_states - 1),
                next_state,
            )
            constraint_reward = torch.where(
                mask,
                constraint_rewards[action_name] * torch.ones_like(constraint_reward),
                constraint_reward,
            )

        done = torch.zeros_like(state, dtype=torch.bool, device=self.device)

        # Big reward for reaching the end pos, additive with normal reward
        normal_reward = torch.where(
            next_state == self.n_states - 1, self.big_reward, normal_reward
        )
        done = torch.where(next_state == self.n_states - 1, True, done)

        reward = (
            (normal_reward + constraint_reward)
            if self.constraints_active
            else normal_reward
        )

        out = TensorDict(
            {
                "observation": next_state.unsqueeze(-1),
                "reward": reward.unsqueeze(-1),
                "normal_reward": normal_reward.unsqueeze(-1),
                "constraint_reward": constraint_reward.unsqueeze(-1),
                "done": done,
            },
            td.shape,
        )
        return out

    @staticmethod
    def calculate_xy(n_states, shortcut_steps, return_x, return_y, big_reward, gamma):
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

DEFAULT_X = 0
DEFAULT_Y = 0

class WorkingEnv(EnvBase):

    batch_locked = True

    def __init__(self, batch_size, seed=None, device="cpu"):

        super().__init__(device=device, batch_size=batch_size)

        self._make_spec()

        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)


    @staticmethod
    def _step(tensordict):
        th, thdot = tensordict["th"], tensordict["thdot"]  # th := theta

        reward = torch.zeros_like(th, dtype=torch.float32).unsqueeze(-1)
        done = torch.zeros_like(th, dtype=torch.bool)
        out = TensorDict(
            {
                "th": th,
                "thdot": thdot,
                "reward": reward,
                "done": done,
            },
            tensordict.shape,
        )
        return out

    def _reset(self, td):
        if td is None:
            batch_size = self.batch_size
        else:
            batch_size = td.shape
        # th = torch.zeros(self.batch_size, 1)
        th = torch.zeros(*batch_size, 1)
        # thdot = torch.zeros(self.batch_size, 1)
        thdot = torch.zeros(*batch_size, 1)
        out = TensorDict(
            {
                "th": th,
                "thdot": thdot,
            },
            # batch_size=self.batch_size
            batch_size=batch_size
        )
        return out

    def _make_spec(self):
        # Under the hood, this will populate self.output_spec["observation"]
        self.observation_spec = Composite(
            th=Bounded(
                low=-torch.pi,
                high=torch.pi,
                shape=(*self.batch_size, 1),
                dtype=torch.float32,
            ),
            thdot=Bounded(
                low=0,
                high=0,
                shape=(*self.batch_size, 1),
                dtype=torch.float32,
            ),
            shape=self.batch_size,
        )
        # since the environment is stateless, we expect the previous output as input.
        # For this, ``EnvBase`` expects some state_spec to be available
        self.state_spec = self.observation_spec.clone()
        # action-spec will be automatically wrapped in input_spec when
        # `self.action_spec = spec` will be called supported
        self.action_spec = Bounded(
            low=0,
            high=0,
            shape=(*self.batch_size,1),
            dtype=torch.float32,
        )
        self.reward_spec = Unbounded(shape=(*self.batch_size,1))

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng
