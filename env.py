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
        n_pos,
        big_reward,
        random_start=False,
        punishment=0,
        seed=None,
        device="cpu",
        constraints_enabled=False,
    ):
        super().__init__(device=device, batch_size=[])

        assert n_pos % 2 == 0, "n_pos only tested for even numbers"
        self.left_reward = left_reward
        self.right_reward = right_reward
        self.down_reward = down_reward
        self.up_reward = up_reward
        self.n_pos = n_pos
        self.big_reward = big_reward
        self.random_start = random_start
        self.punishment = punishment
        self.constraints_enabled = constraints_enabled

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
        self.reward_spec = UnboundedContinuous(shape=(1,), dtype=torch.float32)

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
        # x, y, n_pos, big_reward = self.x, self.y, self.n_pos, self.big_reward

        next_pos = pos.clone()
        reward = 0 * torch.ones_like(pos, dtype=torch.int)

        mask_even = pos % 2 == 0

        # Enable left action by default
        next_pos = torch.where(action == 0, pos - 1, next_pos)
        # Enable right action by default
        next_pos = torch.where(action == 1, pos + 1, next_pos)

        # For even pos, enable down and up actions
        # Down action
        next_pos = torch.where(mask_even & (action == 2), pos - 2, next_pos)
        # Up action
        next_pos = torch.where(mask_even & (action == 3), pos + 2, next_pos)

        if self.constraints_enabled:
            # Left action
            reward = torch.where(action == 0, self.left_reward, reward)
            # Right action
            reward = torch.where(action == 1, self.right_reward, reward)
            # Down action
            reward = torch.where(mask_even & (action == 2), self.down_reward, reward)
            # Up action
            reward = torch.where(mask_even & (action == 3), self.up_reward, reward)

        # Ensure that we can never move past the end pos
        next_pos = torch.where(next_pos >= self.n_pos, self.n_pos - 1, next_pos)

        # Ensure that we can never move before the start pos
        next_pos = torch.where(next_pos < 0, pos, next_pos)

        # If we did not move, terminate the episode and (maybe) punish
        # done = torch.where(next_pos == pos, 1.0, 0.0).to(torch.bool)
        done = torch.zeros_like(pos, dtype=torch.bool)  # TODO
        reward = torch.where(next_pos == pos, -self.punishment, reward)

        # Big reward for reaching the end pos
        reward = torch.where(next_pos == self.n_pos - 1, self.big_reward, reward)
        # If we reach final pos, we're done
        done = torch.where(next_pos == self.n_pos - 1, 1.0, done).to(torch.bool)

        out = TensorDict(
            {
                "pos": next_pos,
                "reward": reward,
                "done": done,
            },
            td.shape,
        )
        return out

    def set_constraint_state(self, constraints_enabled):
        self.constraints_enabled = constraints_enabled


def get_base_env(**kwargs):
    env = BaseEnv(**kwargs)
    env.set_constraint_state(True)
    env = TransformedEnv(env, Compose(StepCounter()))
    check_env_specs(env)
    return env


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
            }
        )

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng

    def _step(self, td):
        # One meta step equals processing one batch of base agent data

        # Apply meta action, which will affect self.base_iter
        self.base_env.set_constraint_state(td["action"].item())

        # Get next base batch and update base agent
        base_td = next(self.base_iter)
        self.base_agent.process_batch(base_td)

        # Next meta state is the mean and std of the base rewards, next meta reward is the sum of the base rewards
        meta_td = TensorDict(
            {
                "state": self._meta_state_from_base_td(base_td),
                "reward": self._meta_reward_from_base_td(base_td),
            },
            batch_size=(),
        )

        return meta_td

    def _make_spec(self):
        self.observation_spec = Composite(
            state=UnboundedContinuous(shape=(2,), dtype=torch.float32),
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
