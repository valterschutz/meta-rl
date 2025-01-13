import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from tensordict.nn.probabilistic import InteractionType
from torch import multiprocessing, nn
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers import (ListStorage, PrioritizedSampler,
                                         ReplayBuffer)
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (Compose, DMControlEnv, DoubleToFloat,
                          ObservationNorm, ParallelEnv, StepCounter,
                          TransformedEnv, set_gym_backend)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.transforms import CatTensors
from torchrl.envs.utils import (ExplorationType, check_env_specs,
                                set_exploration_type)
from torchrl.modules import (OneHotCategorical, ProbabilisticActor, TanhNormal,
                             TruncatedNormal, ValueOperator, MLP)
from torchrl.modules.tensordict_module import SafeModule
from torchrl.objectives import (DDPGLoss, DiscreteSACLoss, SACLoss, SoftUpdate,
                                ValueEstimators)
from torchrl.objectives.value import GAE
from torchrl.record import VideoRecorder
from torchrl.record.loggers.csv import CSVLogger
from tqdm import tqdm

import wandb

sys.path.append(os.path.join(os.path.dirname(__file__), "./"))

from envs.toy_env import ToyEnv
from utils import calc_return


class ToySACAgent:
    def __init__(self, action_spec, state_spec, n_states, device, buffer_size, min_buffer_size, batch_size, sub_batch_size, num_epochs, gamma):
        self.action_spec = action_spec
        self.state_spec = state_spec
        self.device = device
        self.buffer_size = buffer_size
        self.min_buffer_size = min_buffer_size
        self.batch_size = batch_size
        self.sub_batch_size = sub_batch_size
        self.num_epochs = num_epochs

        # Tuned to toy env
        self.actor_lr = 1e-2
        self.critic_lr = 1e-2
        self.alpha_lr = 1e-2
        self.gamma = gamma
        self.target_eps = 0.99
        self.alpha = 0.7
        self.beta = 0.5
        self.max_grad_norm = 100


        n_actions = self.action_spec.shape[-1]

        actor_net = MLP(in_features=1, out_features=n_actions, depth=2, num_cells=n_states, activation_class=nn.ReLU, device=device)

        policy_module = SafeModule(
            module=actor_net, in_keys=["observation"], out_keys=["logits"], spec=self.action_spec
        )

        self.policy_module = ProbabilisticActor(
            module=policy_module,
            spec=self.action_spec,
            in_keys=["logits"],
            distribution_class=OneHotCategorical,
            default_interaction_type=InteractionType.RANDOM,
            return_log_prob=True,
        )

        qvalue_net = MLP(in_features=1, out_features=n_actions, depth=2, num_cells=n_states, activation_class=nn.ReLU, device=device)

        self.qvalue_module = ValueOperator(
            module=qvalue_net,
            in_keys=["observation"],
            out_keys=["action_value"],
        )

        self.loss_module = DiscreteSACLoss(
            actor_network=self.policy_module,
            qvalue_network=self.qvalue_module,
            num_actions=n_actions,
            delay_qvalue=True,
            num_qvalue_nets=2,
            target_entropy_weight=0.2,
            # target_entropy=0.0,
            loss_function="l2"
        )

        self.loss_module.make_value_estimator(ValueEstimators.TD0, gamma=self.gamma)
        self.loss_keys = ["loss_actor", "loss_qvalue", "loss_alpha"]

        self.target_net = SoftUpdate(self.loss_module, eps=self.target_eps)

        self.optimizers = {
            "actor": torch.optim.Adam(
            list(self.loss_module.actor_network_params.flatten_keys().values()),
            lr=self.actor_lr
            ),
            "critic": torch.optim.Adam(
            list(self.loss_module.qvalue_network_params.flatten_keys().values()),
            lr=self.critic_lr
            ),
            "alpha": torch.optim.Adam(
            [self.loss_module.log_alpha],
            lr=self.alpha_lr
            )
            # "all": torch.optim.Adam(self.loss_module.parameters(), lr=self.actor_lr)
        }

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
        loss_dict = {loss_key: [] for loss_key in self.loss_keys}
        grads = []
        alphas = []
        entropy = []
        for _ in range(self.num_epochs):
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

            alphas.append(loss_vals["alpha"].item())
            entropy.append(loss_vals["entropy"].item())

            grad = torch.nn.utils.clip_grad_norm_(self.loss_module.parameters(), self.max_grad_norm)
            grads.append(grad)
            for optim in self.optimizers.values():
                optim.step()
                optim.zero_grad()

            # Update target network
            self.target_net.step()
            self.replay_buffer.update_tensordict_priority(subdata)

        loss_dict = {loss_key: sum(loss_dict[loss_key]) / len(loss_dict[loss_key]) for loss_key in self.loss_keys}
        info_dict = {
           "mean_gradient_norm": sum(grads) / len(grads),
           "mean_alpha": sum(alphas) / len(alphas),
           "mean_entropy": sum(entropy) / len(entropy),
        }

        return loss_dict, info_dict

class PointSACAgent:
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
            # target_entropy=0.0,
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

class ReacherSACAgent:
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
            # target_entropy=0.0,
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
