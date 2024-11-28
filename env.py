from typing import Optional

import torch
import tqdm

from tensordict import TensorDict, TensorDictBase

from torchrl.data import (
    Bounded,
    Composite,
    Unbounded,
    Categorical,
    UnboundedContinuous,
    Binary,
)
from torchrl.envs import (
    EnvBase,
)

from collections import defaultdict
from tqdm import tqdm
from datetime import datetime

from torchrl.envs.utils import (
    check_env_specs,
    step_mdp,
    set_exploration_type,
    ExplorationType,
)
from torchrl.envs.transforms import TransformedEnv, Compose, StepCounter
from torchrl.modules.tensordict_module.actors import QValueActor
from torchrl.modules.tensordict_module import ProbabilisticActor, ValueOperator
from torchrl.modules import EGreedyModule
from torchrl.data import OneHot, ReplayBuffer, PrioritizedReplayBuffer
from torchrl.data.replay_buffers import LazyTensorStorage, SamplerWithoutReplacement
from torchrl.objectives import (
    DQNLoss,
    SoftUpdate,
    ClipPPOLoss,
    ReinforceLoss,
    DiscreteSACLoss,
)
from torchrl.objectives.value import GAE
from torchrl.collectors import SyncDataCollector
from torchrl.trainers import Trainer
from torchrl.record.loggers.csv import CSVLogger
from torchrl.record.loggers.wandb import WandbLogger


from tensordict import TensorDict
from tensordict.nn import TensorDictSequential, TensorDictModule
from tensordict.nn.distributions import (
    NormalParamExtractor,
    OneHotCategorical,
    # Categorical,
)
from torch.distributions import Bernoulli
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import numpy as np

import wandb


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


class BaseEnv(EnvBase):
    batch_locked = False

    def __init__(
        self,
        left_reward,
        right_reward,
        down_reward,
        up_reward,
        n_states,
        big_reward,
        random_start=False,
        punishment=0,
        seed=None,
        device="cpu",
        constraints_enabled=False,
        left_weight=1.0,
        right_weight=1.0,
        down_weight=1.0,
        up_weight=1.0,
    ):
        super().__init__(device=device, batch_size=[])

        assert n_states % 2 == 0, "n_states only tested for even numbers"
        self.left_reward = left_reward
        self.right_reward = right_reward
        self.down_reward = down_reward
        self.up_reward = up_reward
        self.n_states = n_states
        self.big_reward = big_reward
        self.random_start = random_start
        self.punishment = punishment
        self.constraints_enabled = constraints_enabled
        self.left_weight = left_weight
        self.right_weight = right_weight
        self.down_weight = down_weight
        self.up_weight = up_weight

        self._make_spec()
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    def _make_spec(self):
        self.observation_spec = Composite(
            state=Categorical(self.n_states, shape=(), dtype=torch.int32),
            shape=(),
        )
        self.state_spec = Composite(
            state=Categorical(self.n_states, shape=(), dtype=torch.int32), shape=()
        )
        self.action_spec = Categorical(4, shape=(), dtype=torch.int32)
        self.reward_spec = UnboundedContinuous(shape=(1,), dtype=torch.float32)

    def _reset(self, td):
        if td is None or td.is_empty():
            shape = ()
        else:
            shape = td.shape

        if self.random_start:
            state = torch.randint(
                0, self.n_states, shape=shape, dtype=torch.int32, device=self.device
            )
        else:
            state = torch.zeros(shape, dtype=torch.int32, device=self.device)

        out = TensorDict(
            {
                "state": state,
            },
            batch_size=shape,
        )
        return out

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng

    def _step(self, td):
        state = td["state"]
        action = td["action"]  # Action order: left, right, down, up
        # x, y, n_pos, big_reward = self.x, self.y, self.n_pos, self.big_reward

        next_state = state.clone()
        reward = 0 * torch.ones_like(state, dtype=torch.int)

        mask_even = state % 2 == 0

        # Enable left action by default
        next_state = torch.where(action == 0, state - 1, next_state)
        # Enable right action by default
        next_state = torch.where(action == 1, state + 1, next_state)

        # For even pos, enable down and up actions
        # Down action
        next_state = torch.where(mask_even & (action == 2), state - 2, next_state)
        # Up action
        next_state = torch.where(mask_even & (action == 3), state + 2, next_state)

        # print(f"Constraints enabled: {self.constraints_enabled}")
        if self.constraints_enabled:
            # print(f"Constraints enabled")
            # Left action
            reward = torch.where(
                action == 0, self.left_weight * self.left_reward, reward
            )
            # Right action
            reward = torch.where(
                action == 1, self.right_weight * self.right_reward, reward
            )
            # Down action
            reward = torch.where(
                mask_even & (action == 2), self.down_weight * self.down_reward, reward
            )
            # Up action
            reward = torch.where(
                mask_even & (action == 3), self.up_weight * self.up_reward, reward
            )
        # else:
        # print(f"Constraints disabled")

        # Ensure that we can never move past the end pos
        next_state = torch.where(
            next_state >= self.n_states, self.n_states - 1, next_state
        )

        # Ensure that we can never move before the start pos
        next_state = torch.where(next_state < 0, state, next_state)

        # Punish for moving to the same pos
        done = torch.zeros_like(state, dtype=torch.bool)
        reward = torch.where(next_state == state, -self.punishment, reward)

        # Big reward for reaching the end pos, overriding the possible constraints
        reward = torch.where(next_state == self.n_states - 1, self.big_reward, reward)
        # If we reach final pos, we're done
        done = torch.where(next_state == self.n_states - 1, 1.0, done).to(torch.bool)

        out = TensorDict(
            {
                "state": next_state,
                "reward": reward,
                "done": done,
            },
            td.shape,
        )
        return out

    def set_constraint_state(self, constraints_enabled):
        self.constraints_enabled = constraints_enabled

    @staticmethod
    def calculate_xy(n_states, return_x, return_y, big_reward, gamma):
        # Assuming n_pos is even, calculate x and y
        assert n_states % 2 == 0
        nx = n_states - 2
        ny = (n_states - 2) // 2
        x = (return_x - big_reward * gamma**nx) / sum(gamma**k for k in range(0, nx))
        y = (return_y - big_reward * gamma**ny) / sum(gamma**k for k in range(0, ny))
        return x, y

    @classmethod
    def get_base_env(cls, **kwargs):
        env = cls(**kwargs)
        env.constraints_enabled = kwargs["constraints_enabled"]
        env = TransformedEnv(env, Compose(StepCounter()))
        check_env_specs(env)
        return env

    def set_left_weight(self, left_weight):
        self.left_weight = left_weight

    def set_right_weight(self, right_weight):
        self.right_weight = right_weight

    def set_down_weight(self, down_weight):
        self.down_weight = down_weight

    def set_up_weight(self, up_weight):
        self.up_weight = up_weight

    def set_constraint_weight(self, weight):
        self.set_left_weight(weight)
        self.set_right_weight(weight)
        self.set_down_weight(weight)
        self.set_up_weight(weight)


class MetaEnv(EnvBase):
    def __init__(self, base_env, base_agent, total_frames, device, seed=None):
        super().__init__(device=device, batch_size=[])
        self.base_env = base_env
        self.base_agent = base_agent

        self.base_collector = SyncDataCollector(
            self.base_env,
            self.base_agent.policy,
            frames_per_batch=self.base_agent.buffer_size,
            total_frames=-1,
            split_trajs=False,
            device="cpu",
        )
        self.base_iter = iter(self.base_collector)

        self._make_spec()
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    def _reset(self, td):
        # Reset the base agent and return the initial state for the meta agent
        self.base_agent.reset()

        return TensorDict(
            {
                "state": torch.tensor([0.0, 0.0]),
                "base_agent_losses": TensorDict(
                    {
                        "loss_objective": torch.tensor(
                            0.0, device=self.device, dtype=torch.float32
                        ),
                        "loss_critic": torch.tensor(
                            0.0, device=self.device, dtype=torch.float32
                        ),
                        "loss_entropy": torch.tensor(
                            0.0, device=self.device, dtype=torch.float32
                        ),
                    },
                    batch_size=(),
                ),
                "base_agent_grad_norm": torch.tensor(0.0),
            }
        )

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng

    def _step(self, td):
        # One meta step equals processing one batch of base agent data

        # Apply meta action, which will affect self.base_iter
        # self.base_env.set_constraint_state(td["action"].item()) # TODO: uncomment once base agent loss converges

        # Get next base batch and update base agent
        base_td = next(self.base_iter)
        base_agent_losses, base_agent_grad_norm = self.base_agent.process_batch(base_td)

        # Next meta state is the mean and std of the base rewards, next meta reward is the sum of the base rewards
        meta_td = TensorDict(
            {
                "state": self._meta_state_from_base_td(base_td),
                "reward": self._meta_reward_from_base_td(base_td),
                "base_agent_losses": base_agent_losses,
                "base_agent_grad_norm": base_agent_grad_norm,
            },
            batch_size=(),
        )

        return meta_td

    def _make_spec(self):
        self.observation_spec = Composite(
            state=UnboundedContinuous(shape=(2,), dtype=torch.float32),
            base_agent_losses=Composite(
                loss_objective=UnboundedContinuous(shape=(), dtype=torch.float32),
                loss_critic=UnboundedContinuous(shape=(), dtype=torch.float32),
                loss_entropy=UnboundedContinuous(shape=(), dtype=torch.float32),
            ),
            base_agent_grad_norm=UnboundedContinuous(shape=(), dtype=torch.float32),
            shape=(),
        )
        self.state_spec = Composite(
            state=UnboundedContinuous(shape=(2,), dtype=torch.float32),
        )
        self.action_spec = Binary(shape=(1,), dtype=torch.bool)
        self.reward_spec = UnboundedContinuous(shape=(1,), dtype=torch.float32)

    @staticmethod
    def _meta_state_from_base_td(base_td):
        # Note the use of .detach() to avoid backpropagating through the base agent
        return torch.tensor(
            [
                base_td["next", "reward"].mean().detach(),
                base_td["next", "reward"].std().detach(),
            ]
        )

    @staticmethod
    def _meta_reward_from_base_td(base_td):
        # Note the use of .detach() to avoid backpropagating through the base agent
        return base_td["next", "reward"].sum().detach()
