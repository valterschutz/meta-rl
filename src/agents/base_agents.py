import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from tensordict.nn.probabilistic import InteractionType
from torch import multiprocessing, nn
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (Compose, DMControlEnv, DoubleToFloat,
                          ObservationNorm, StepCounter, TransformedEnv,
                          set_gym_backend)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.transforms import CatTensors
from torchrl.envs import ParallelEnv
from torchrl.envs.utils import (ExplorationType, check_env_specs,
                                set_exploration_type)
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import DiscreteSACLoss, SoftUpdate, SACLoss
from torchrl.objectives.value import GAE
from torchrl.record import VideoRecorder
from torchrl.record.loggers.csv import CSVLogger
from torchrl.modules import OneHotCategorical, TruncatedNormal
from tqdm import tqdm
from torchrl.objectives import ValueEstimators
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers import PrioritizedSampler, ListStorage

import wandb

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "./"))

from envs.toy_env import ToyEnv
from utils import calc_return

class ToyBaseAgent:
    def __init__(self, action_spec, state_spec, device, buffer_size, min_buffer_size, batch_size, sub_batch_size, num_epochs, lr, gamma, target_eps, alpha, beta, max_grad_norm):
        self.action_spec = action_spec
        self.state_spec = state_spec
        self.device = device
        self.buffer_size = buffer_size
        self.min_buffer_size = min_buffer_size
        self.batch_size = batch_size
        self.sub_batch_size = sub_batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.gamma = gamma
        self.target_eps = target_eps
        self.alpha = alpha
        self.beta = beta
        self.max_grad_norm = max_grad_norm

        n_states = self.state_spec["state"].shape[-1]
        n_actions = self.action_spec.shape[-1]
        hidden_units = 1
        actor_net = nn.Sequential(
            nn.Linear(n_states, hidden_units, device=device),
            nn.Linear(hidden_units, n_actions, device=device),
        )

        policy_module = TensorDictModule(
            actor_net, in_keys=["state"], out_keys=["logits"]
        )

        self.policy_module = ProbabilisticActor(
            module=policy_module,
            spec=self.action_spec,
            in_keys=["logits"],
            distribution_class=OneHotCategorical,
            default_interaction_type=InteractionType.RANDOM, # TODO: should this be random?
            return_log_prob=True,
        )

        qvalue_net = nn.Sequential(
            nn.Linear(n_states, hidden_units, device=device),
            nn.Linear(hidden_units, n_actions, device=device),
        )
        self.qvalue_module = ValueOperator(
            module=qvalue_net,
            in_keys=["state"],
            out_keys=["action_value"],
        )

        self.loss_module = DiscreteSACLoss(
            actor_network=self.policy_module,
            qvalue_network=self.qvalue_module,
            num_actions=n_actions,
            # alpha_init=1e-3,
            # max_alpha=1e-3,
            target_entropy=0.0,
            # alpha_init=0.1,
        )

        self.loss_module.make_value_estimator(ValueEstimators.TD0, gamma=self.gamma)
        self.loss_keys = ["loss_actor", "loss_qvalue", "loss_alpha"]

        self.target_net = SoftUpdate(self.loss_module, eps=self.target_eps)

        self.optim = torch.optim.Adam(self.loss_module.parameters(), self.lr)

        self.replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(max_size=self.buffer_size, device=self.device),
            sampler=PrioritizedSampler(max_capacity=self.buffer_size, alpha=self.alpha, beta=self.beta),
            priority_key="td_error",
            batch_size=self.batch_size,
        )


    def process_batch(self, td, constraints_active):
        # Add data to the replay buffer
        data_view = td.reshape(-1)
        self.replay_buffer.extend(data_view.to(self.device))
        if len(self.replay_buffer) < self.min_buffer_size:
            return
        for _ in range(self.num_epochs):
            loss_dict = {loss_key: [] for loss_key in self.loss_keys}
            grads = []

            subdata = self.replay_buffer.sample(self.sub_batch_size)
            # Interpret rewards differently depending on the batch
            if constraints_active:
                subdata["next", "reward"] = subdata["next", "normal_reward"] + subdata["next", "constraint_reward"]
            else:
                subdata["next", "reward"] = subdata["next", "normal_reward"]
            loss_vals = self.loss_module(subdata.to(self.device))
            loss = 0
            for loss_key in self.loss_keys:
                loss += loss_vals[loss_key]
                loss_dict[loss_key].append(loss_vals[loss_key].item())

            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(self.loss_module.parameters(), self.max_grad_norm)
            grads.append(grad)
            self.optim.step()
            self.optim.zero_grad()

            # Update target network
            self.target_net.step()
            self.replay_buffer.update_tensordict_priority(subdata)

        loss_dict = {loss_key: sum(loss_dict[loss_key]) / len(loss_dict[loss_key]) for loss_key in self.loss_keys}
        info_dict = {
           "mean_gradient_norm": sum(grads) / len(grads),
        }

        return loss_dict, info_dict

class PointBaseAgent:
    def __init__(self, action_spec, device, buffer_size, min_buffer_size, batch_size, sub_batch_size, num_epochs, lr, gamma, target_eps, alpha, beta, max_grad_norm):
        self.action_spec = action_spec
        self.device = device
        self.buffer_size = buffer_size
        self.min_buffer_size = min_buffer_size
        self.batch_size = batch_size
        self.sub_batch_size = sub_batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.gamma = gamma
        self.target_eps = target_eps
        self.alpha = alpha
        self.beta = beta
        self.max_grad_norm = max_grad_norm

        n_states = 4
        n_actions = 2
        hidden_units = 8
        actor_net = nn.Sequential(
            nn.Linear(n_states, hidden_units, device=device),
            nn.ReLU(),
            nn.Linear(hidden_units, 2*n_actions, device=device),
            NormalParamExtractor()
        )

        policy_module = TensorDictModule(
            actor_net, in_keys=["state"], out_keys=["loc", "scale"]
        )

        self.policy_module = ProbabilisticActor(
            module=policy_module,
            spec=self.action_spec,
            in_keys=["loc", "scale"],
            distribution_class=TruncatedNormal,
            default_interaction_type=InteractionType.RANDOM,
            return_log_prob=True,
        )

        class QValueNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(n_states+n_actions, hidden_units, device=device),
                    nn.ReLU(),
                    nn.Linear(hidden_units, 1, device=device),
                )

            def forward(self, state, action):
                return self.net(torch.cat([state, action], -1))
        self.qvalue_module = ValueOperator(
            module=QValueNet(),
            in_keys=["state", "action"],
            out_keys=["state_action_value"],
        )

        self.loss_module = SACLoss(
            actor_network=self.policy_module,
            qvalue_network=self.qvalue_module,
            # alpha_init=1e-3,
            # max_alpha=1e-3,
            target_entropy=0.0,
            # alpha_init=0.1,
        )

        self.loss_module.make_value_estimator(ValueEstimators.TD0, gamma=self.gamma)
        self.loss_keys = ["loss_actor", "loss_qvalue", "loss_alpha"]

        self.target_net = SoftUpdate(self.loss_module, eps=self.target_eps)

        self.optim = torch.optim.Adam(self.loss_module.parameters(), self.lr)

        self.replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(max_size=self.buffer_size, device=self.device),
            sampler=PrioritizedSampler(max_capacity=self.buffer_size, alpha=self.alpha, beta=self.beta),
            priority_key="td_error",
            batch_size=self.batch_size,
        )


    def process_batch(self, td, constraints_active):
        # Add data to the replay buffer
        data_view = td.reshape(-1)
        self.replay_buffer.extend(data_view.to(self.device))
        if len(self.replay_buffer) < self.min_buffer_size:
            return
        for _ in range(self.num_epochs):
            loss_dict = {loss_key: [] for loss_key in self.loss_keys}
            grads = []

            subdata = self.replay_buffer.sample(self.sub_batch_size)
            # Interpret rewards differently depending on the batch
            if constraints_active:
                subdata["next", "reward"] = subdata["next", "normal_reward"] + subdata["next", "constraint_reward"]
            else:
                subdata["next", "reward"] = subdata["next", "normal_reward"]
            loss_vals = self.loss_module(subdata.to(self.device))
            loss = 0
            for loss_key in self.loss_keys:
                loss += loss_vals[loss_key]
                loss_dict[loss_key].append(loss_vals[loss_key].item())

            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(self.loss_module.parameters(), self.max_grad_norm)
            grads.append(grad)
            self.optim.step()
            self.optim.zero_grad()

            # Update target network
            self.target_net.step()
            self.replay_buffer.update_tensordict_priority(subdata)

        loss_dict = {loss_key: sum(loss_dict[loss_key]) / len(loss_dict[loss_key]) for loss_key in self.loss_keys}
        info_dict = {
           "mean_gradient_norm": sum(grads) / len(grads),
        }

        return loss_dict, info_dict

def slow_policy(td):
    # Always go right
    td["action"] = torch.tensor([0, 1, 0, 0], device=td.device)
    return td


def fast_policy(td):
    # Always go up in even states, otherwise go right
    state_idx = td["state"].argmax(dim=-1)
    if state_idx % 2 == 0:
        td["action"] = torch.tensor([0, 0, 0, 1], device=td.device)
    else:
        td["action"] = torch.tensor([0, 1, 0, 0], device=td.device)
    return td
