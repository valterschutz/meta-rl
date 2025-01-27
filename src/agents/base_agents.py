import math
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import os
import sys
from abc import ABC, abstractmethod
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from tensordict.nn import TensorDictModule, TensorDictSequential
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
from torchrl.modules import (MLP, EGreedyModule, OneHotCategorical,
                             ProbabilisticActor, TanhNormal, TruncatedNormal,
                             ValueOperator)
from torchrl.modules.tensordict_module import Actor, QValueActor, SafeModule
from torchrl.objectives import (DDPGLoss, DiscreteSACLoss, DQNLoss, SACLoss,
                                SoftUpdate, TD3Loss, ValueEstimators)
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


    # def __init__(self, *args):
    #     super().__init__(*args)




class OffpolicyAgent(ABC):
    @abstractmethod
    def get_agent_details(self, agent_detail_args):
        """
        Should return a loss module, a list of loss keys, a list of optimizers, a dictionary specifying maximum gradients, and the policy.
        """
        pass

    @abstractmethod
    def update_callback(self):
        """
        Called after each optimization step in `process_batch`
        """
        pass

    @abstractmethod
    def train_info_dict_callback(self, td):
        """
        Called once at the end of `process_batch`. Should return a dictionary of information to pass to `wandb.log`.
        """
        pass

    @abstractmethod
    def eval_info_dict_callback(self, td):
        """
        Called during evaluation. Should return a dictionary of information to pass to `wandb.log`.
        """
        pass


    def __init__(self, device, batch_size, sub_batch_size, num_epochs, replay_buffer_args, env, agent_detail_args):
        self.device = device
        self.sub_batch_size = sub_batch_size
        self.num_epochs = num_epochs
        self.env = env

        # Subclasses implement this (template method pattern)
        self.loss_module, self.loss_keys, self.optims, self.max_grad_dict, self.train_policy, self.eval_policy = self.get_agent_details(agent_detail_args)

        self.replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(max_size=replay_buffer_args["buffer_size"], device=self.device),
            sampler=PrioritizedSampler(max_capacity=replay_buffer_args["buffer_size"], alpha=replay_buffer_args["alpha"], beta=replay_buffer_args["beta"]),
            priority_key="td_error",
            batch_size=batch_size,
        )
        self.min_buffer_size = replay_buffer_args["min_buffer_size"]


    def process_batch(self, td, constraints_active):
        # Add data to the replay buffer
        data_view = td.reshape(-1)
        self.replay_buffer.extend(data_view.to(self.device))
        if len(self.replay_buffer) < self.min_buffer_size:
            return
        for _ in range(self.num_epochs):
            loss_dict = {loss_key: [] for loss_key in self.loss_keys}
            pre_clip_grads = defaultdict(list)
            post_clip_grads = defaultdict(list)

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

            for k, v in self.max_grad_dict.items():
                params_fn, max_grad_norm = v
                grad = torch.nn.utils.clip_grad_norm_(params_fn(), max_grad_norm)
                pre_clip_grads[k].append(grad)
                post_clip_norm = torch.norm(
                    torch.stack([
                        p.grad.norm() for p in params_fn() if p.grad is not None
                    ])
                )
                post_clip_grads[k].append(post_clip_norm)

            for optim in self.optims:
                optim.step()
                optim.zero_grad()

            self.update_callback()
            self.replay_buffer.update_tensordict_priority(subdata)

        loss_dict = {loss_key: sum(loss_dict[loss_key]) / len(loss_dict[loss_key]) for loss_key in self.loss_keys}
        grad_dict = {}
        for k, v in pre_clip_grads.items():
            grad_dict[f"pre-clip mean_{k}_gradient_norm"] = sum(v) / len(v)
        for k, v in post_clip_grads.items():
            grad_dict[f"post-clip mean_{k}_gradient_norm"] = sum(v) / len(v)

        return loss_dict, grad_dict

class ReacherSACAgent(OffpolicyAgent):
    #override
    def get_agent_details(self, agent_detail_args):
        hidden_units = agent_detail_args["hidden_units"]
        lr = agent_detail_args["lr"]
        gamma = agent_detail_args["agent_gamma"]
        target_eps = agent_detail_args["target_eps"]
        actor_max_grad = agent_detail_args["actor_max_grad"]
        value_max_grad = agent_detail_args["value_max_grad"]

        n_states = 4
        n_actions = 2
        actor_net = nn.Sequential(
            nn.Linear(n_states, hidden_units, device=self.device),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units, device=self.device),
            nn.ReLU(),
            nn.Linear(hidden_units, 2*n_actions, device=self.device),
            NormalParamExtractor()
        )

        policy_module = TensorDictModule(
            actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
        )

        policy_module = ProbabilisticActor(
            module=policy_module,
            spec=self.env.action_spec,
            in_keys=["loc", "scale"],
            distribution_class=TruncatedNormal,
            default_interaction_type=InteractionType.RANDOM,
            return_log_prob=True,
        )

        class QValueNet(nn.Module):
            def __init__(self, device):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(n_states+n_actions, hidden_units, device=device),
                    nn.ReLU(),
                    nn.Linear(hidden_units, hidden_units, device=device),
                    nn.ReLU(),
                    nn.Linear(hidden_units, 1, device=device),
                )

            def forward(self, state, action):
                return self.net(torch.cat([state, action], -1))

        qvalue_module = ValueOperator(
            module=QValueNet(device=self.device),
            in_keys=["observation", "action"],
            out_keys=["state_action_value"],
        )

        loss_module = SACLoss(
            actor_network=policy_module,
            qvalue_network=qvalue_module,
        )

        loss_module.make_value_estimator(ValueEstimators.TD0, gamma=gamma)
        loss_keys = ["loss_actor", "loss_qvalue", "loss_alpha"]

        optim = torch.optim.Adam(loss_module.parameters(), lr=lr)

        self.target_net = SoftUpdate(loss_module, eps=target_eps)

        max_grad_dict = {
            "actor": (loss_module.actor_network_params.parameters, actor_max_grad),
            "value": (loss_module.qvalue_network_params.parameters, value_max_grad)
        }

        return loss_module, loss_keys, [optim], max_grad_dict, policy_module

    #override
    def update_callback(self):
        self.target_net.step()

    #override
    def info_dict_callback(self, td, loss_module):
        return  {
            "pos x distribution": wandb.Histogram(td["position"][0].cpu()),
            "pos y distribution": wandb.Histogram(td["position"][1].cpu()),
            "vel x distribution": wandb.Histogram(td["velocity"][0].cpu()),
            "vel y distribution": wandb.Histogram(td["velocity"][1].cpu()),
            "x action": wandb.Histogram(td["action"][0].cpu()),
            "y action": wandb.Histogram(td["action"][1].cpu()),
            "policy 'norm'": sum((p**2).sum() for p in loss_module.actor_network_params.parameters()),
        }

class ReacherDDPGAgent(OffpolicyAgent):
    #override
    def get_agent_details(self, agent_detail_args):
        gamma = agent_detail_args["agent_gamma"]
        target_eps = agent_detail_args["target_eps"]
        actor_max_grad = agent_detail_args["actor_max_grad"]
        value_max_grad = agent_detail_args["value_max_grad"]
        actor_lr = agent_detail_args["actor_lr"]
        value_lr = agent_detail_args["value_lr"]

        n_states = 4
        n_actions = 2
        actor_net = nn.Sequential(
            nn.Linear(n_states, 300, device=self.device),
            nn.ReLU(),
            nn.Linear(300, 200, device=self.device),
            nn.ReLU(),
            nn.Linear(200, n_actions, device=self.device),
        )

        policy_module = Actor(
            module=actor_net,
            spec=self.env.action_spec,
            in_keys=["observation"],
        )

        class QValueNet(nn.Module):
            def __init__(self, device):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(n_states+n_actions, 400, device=device),
                    nn.ReLU(),
                    nn.Linear(400, 300, device=device),
                    nn.ReLU(),
                    nn.Linear(300, 1, device=device),
                )

            def forward(self, state, action):
                return self.net(torch.cat([state, action], -1))

        qvalue_module = ValueOperator(
            module=QValueNet(device=self.device),
            in_keys=["observation", "action"],
            out_keys=["state_action_value"],
        )

        loss_module = DDPGLoss(
            actor_network=policy_module,
            value_network=qvalue_module,
        )

        loss_module.make_value_estimator(ValueEstimators.TD0, gamma=gamma)
        loss_keys = ["loss_actor", "loss_value"]

        actor_optim = torch.optim.Adam(loss_module.actor_network_params.parameters(), lr=actor_lr)
        value_optim = torch.optim.Adam(loss_module.value_network_params.parameters(), lr=value_lr)

        self.target_net = SoftUpdate(loss_module, eps=target_eps)

        max_grad_dict = {
            "actor": (loss_module.actor_network_params.parameters, actor_max_grad),
            "value": (loss_module.value_network_params.parameters, value_max_grad)
        }

        return loss_module, loss_keys, [actor_optim, value_optim], max_grad_dict, policy_module

    #override
    def update_callback(self):
        self.target_net.step()

    #override
    def info_dict_callback(self, td, loss_module):
        return  {
            # "pos x distribution": wandb.Histogram(td["position"][0].cpu()),
            # "pos y distribution": wandb.Histogram(td["position"][1].cpu()),
            # "vel x distribution": wandb.Histogram(td["velocity"][0].cpu()),
            # "vel y distribution": wandb.Histogram(td["velocity"][1].cpu()),
            # "x action": wandb.Histogram(td["action"][0].cpu()),
            # "y action": wandb.Histogram(td["action"][1].cpu()),
            "policy 'norm'": sum((p**2).sum() for p in loss_module.actor_network_params.parameters()),
            "value 'norm'": sum((p**2).sum() for p in loss_module.value_network_params.parameters()),
        }

class CartpoleDDPGAgent(OffpolicyAgent):
    #override
    def get_agent_details(self, agent_detail_args):
        gamma = agent_detail_args["agent_gamma"]
        target_eps = agent_detail_args["target_eps"]
        actor_max_grad = agent_detail_args["actor_max_grad"]
        value_max_grad = agent_detail_args["value_max_grad"]
        actor_lr = agent_detail_args["actor_lr"]
        value_lr = agent_detail_args["value_lr"]

        n_states = 5
        n_actions = 1
        actor_net = nn.Sequential(
            nn.Linear(n_states, 300, device=self.device),
            nn.ReLU(),
            nn.Linear(300, 200, device=self.device),
            nn.ReLU(),
            nn.Linear(200, n_actions, device=self.device),
        )

        policy_module = Actor(
            module=actor_net,
            spec=self.env.action_spec,
            in_keys=["observation"],
        )

        class QValueNet(nn.Module):
            def __init__(self, device):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(n_states+n_actions, 400, device=device),
                    nn.ReLU(),
                    nn.Linear(400, 300, device=device),
                    nn.ReLU(),
                    nn.Linear(300, 1, device=device),
                )

            def forward(self, state, action):
                return self.net(torch.cat([state, action], -1))

        qvalue_module = ValueOperator(
            module=QValueNet(device=self.device),
            in_keys=["observation", "action"],
            out_keys=["state_action_value"],
        )

        loss_module = DDPGLoss(
            actor_network=policy_module,
            value_network=qvalue_module,
        )

        loss_module.make_value_estimator(ValueEstimators.TD0, gamma=gamma)
        loss_keys = ["loss_actor", "loss_value"]

        actor_optim = torch.optim.Adam(loss_module.actor_network_params.parameters(), lr=actor_lr)
        value_optim = torch.optim.Adam(loss_module.value_network_params.parameters(), lr=value_lr)

        self.target_net = SoftUpdate(loss_module, eps=target_eps)

        max_grad_dict = {
            "actor": (loss_module.actor_network_params.parameters, actor_max_grad),
            "value": (loss_module.value_network_params.parameters, value_max_grad)
        }

        return loss_module, loss_keys, [actor_optim, value_optim], max_grad_dict, policy_module

    #override
    def update_callback(self):
        self.target_net.step()

    #override
    def info_dict_callback(self, td, loss_module):
        return  {
            # "pos x distribution": wandb.Histogram(td["position"][0].cpu()),
            # "pos y distribution": wandb.Histogram(td["position"][1].cpu()),
            # "vel x distribution": wandb.Histogram(td["velocity"][0].cpu()),
            # "vel y distribution": wandb.Histogram(td["velocity"][1].cpu()),
            # "x action": wandb.Histogram(td["action"][0].cpu()),
            # "y action": wandb.Histogram(td["action"][1].cpu()),
            "policy 'norm'": sum((p**2).sum() for p in loss_module.actor_network_params.parameters()),
            "value 'norm'": sum((p**2).sum() for p in loss_module.value_network_params.parameters()),
        }

class CartpoleTD3Agent(OffpolicyAgent):
    #override
    def get_agent_details(self, agent_detail_args):
        gamma = agent_detail_args["agent_gamma"]
        target_eps = agent_detail_args["target_eps"]
        actor_max_grad = agent_detail_args["actor_max_grad"]
        value_max_grad = agent_detail_args["value_max_grad"]
        actor_lr = agent_detail_args["actor_lr"]
        value_lr = agent_detail_args["value_lr"]
        action_spec = self.env.action_spec

        n_states = 5
        n_actions = 1
        actor_net = nn.Sequential(
            nn.Linear(n_states, 300, device=self.device),
            nn.ReLU(),
            nn.Linear(300, 200, device=self.device),
            nn.ReLU(),
            nn.Linear(200, n_actions, device=self.device),
        )

        policy_module = Actor(
            module=actor_net,
            spec=self.env.action_spec,
            in_keys=["observation"],
        )

        class QValueNet(nn.Module):
            def __init__(self, device):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(n_states+n_actions, 400, device=device),
                    nn.ReLU(),
                    nn.Linear(400, 300, device=device),
                    nn.ReLU(),
                    nn.Linear(300, 1, device=device),
                )

            def forward(self, state, action):
                return self.net(torch.cat([state, action], -1))

        qvalue_module = ValueOperator(
            module=QValueNet(device=self.device),
            in_keys=["observation", "action"],
            out_keys=["state_action_value"],
        )

        loss_module = TD3Loss(
            actor_network=policy_module,
            qvalue_network=qvalue_module,
            action_spec=action_spec
        )

        loss_module.make_value_estimator(ValueEstimators.TD0, gamma=gamma)
        loss_keys = ["loss_actor", "loss_qvalue"]

        actor_optim = torch.optim.Adam(loss_module.actor_network_params.parameters(), lr=actor_lr)
        value_optim = torch.optim.Adam(loss_module.qvalue_network_params.parameters(), lr=value_lr)

        self.target_net = SoftUpdate(loss_module, eps=target_eps)

        max_grad_dict = {
            "actor": (loss_module.actor_network_params.parameters, actor_max_grad),
            "value": (loss_module.qvalue_network_params.parameters, value_max_grad)
        }

        return loss_module, loss_keys, [actor_optim, value_optim], max_grad_dict, policy_module

    #override
    def update_callback(self):
        self.target_net.step()

    #override
    def info_dict_callback(self, td, loss_module):
        return  {
            "policy 'norm'": sum((p**2).sum() for p in loss_module.actor_network_params.parameters()),
            "value 'norm'": sum((p**2).sum() for p in loss_module.qvalue_network_params.parameters()),
        }

class FingerspinTD3Agent(OffpolicyAgent):
    #override
    def get_agent_details(self, agent_detail_args):
        gamma = agent_detail_args["agent_gamma"]
        target_eps = agent_detail_args["target_eps"]
        actor_max_grad = agent_detail_args["actor_max_grad"]
        value_max_grad = agent_detail_args["value_max_grad"]
        actor_lr = agent_detail_args["actor_lr"]
        value_lr = agent_detail_args["value_lr"]
        action_spec = self.env.action_spec

        n_states = 7
        n_actions = 2
        actor_net = nn.Sequential(
            nn.Linear(n_states, 300, device=self.device),
            nn.ReLU(),
            nn.Linear(300, 200, device=self.device),
            nn.ReLU(),
            nn.Linear(200, n_actions, device=self.device),
        )

        policy_module = Actor(
            module=actor_net,
            spec=self.env.action_spec,
            in_keys=["observation"],
        )

        class QValueNet(nn.Module):
            def __init__(self, device):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(n_states+n_actions, 400, device=device),
                    nn.ReLU(),
                    nn.Linear(400, 300, device=device),
                    nn.ReLU(),
                    nn.Linear(300, 1, device=device),
                )

            def forward(self, state, action):
                return self.net(torch.cat([state, action], -1))

        qvalue_module = ValueOperator(
            module=QValueNet(device=self.device),
            in_keys=["observation", "action"],
            out_keys=["state_action_value"],
        )

        loss_module = TD3Loss(
            actor_network=policy_module,
            qvalue_network=qvalue_module,
            action_spec=action_spec
        )

        loss_module.make_value_estimator(ValueEstimators.TD0, gamma=gamma)
        loss_keys = ["loss_actor", "loss_qvalue"]

        actor_optim = torch.optim.Adam(loss_module.actor_network_params.parameters(), lr=actor_lr)
        value_optim = torch.optim.Adam(loss_module.qvalue_network_params.parameters(), lr=value_lr)

        self.target_net = SoftUpdate(loss_module, eps=target_eps)

        max_grad_dict = {
            "actor": (loss_module.actor_network_params.parameters, actor_max_grad),
            "value": (loss_module.qvalue_network_params.parameters, value_max_grad)
        }

        return loss_module, loss_keys, [actor_optim, value_optim], max_grad_dict, policy_module, policy_module

    #override
    def update_callback(self):
        self.target_net.step()

    #override
    def info_dict_callback(self, td, loss_module):
        return  {
            "policy 'norm'": sum((p**2).sum() for p in loss_module.actor_network_params.parameters()),
            "value 'norm'": sum((p**2).sum() for p in loss_module.qvalue_network_params.parameters()),
        }

class ToyDQNAgent(OffpolicyAgent):

    #override
    def get_agent_details(self, agent_detail_args):
        num_cells = agent_detail_args["num_cells"]
        value_lr = agent_detail_args["value_lr"]
        target_eps = agent_detail_args["target_eps"]
        value_max_grad = agent_detail_args["value_max_grad"]
        qvalue_eps = agent_detail_args["qvalue_eps"]

        qvalue_network = MLP(in_features=1, out_features=4, num_cells=num_cells, activation_class=nn.ReLU).to(self.device)
        eval_policy_module = QValueActor(qvalue_network, in_keys=["observation"], spec=self.env.action_spec).to(self.device)
        loss_module = DQNLoss(value_network=eval_policy_module, action_space="one-hot", double_dqn=True)
        optim = torch.optim.Adam(loss_module.parameters(), lr=value_lr)

        self.target_net = SoftUpdate(loss_module, eps=target_eps)

        max_grad_dict = {
            "value": (loss_module.value_network_params.parameters, value_max_grad)
        }

        train_policy_module = TensorDictSequential(
            eval_policy_module,
            EGreedyModule(spec=self.env.action_spec, eps_init=qvalue_eps, eps_end=qvalue_eps)
        )

        return loss_module, ["loss"], [optim], max_grad_dict, train_policy_module, eval_policy_module

    #override
    def update_callback(self):
        self.target_net.step()


    #override
    def info_dict_callback(self, td):
        return {
            "state distribution": wandb.Histogram(td["observation"].cpu()),
            "action distribution": wandb.Histogram(td["action"].argmax(dim=-1).cpu()),
        }

class ToyTabularDQNAgent(OffpolicyAgent):
    def get_policy_matrix(self):
        policy_matrix = torch.zeros((self.n_states, 4))
        for i in range(self.n_states):
            state = torch.tensor([[i]])
            chosen_action = self.eval_policy(state)[0].argmax()
            # The non-chosen actions get weight epsilon, and the chosen action gets weight 1 - 3*epsilon/4
            policy_matrix[i] = torch.tensor([self.qvalue_eps/4, self.qvalue_eps/4, self.qvalue_eps/4, self.qvalue_eps/4])
            policy_matrix[i, chosen_action] = 1 - 3*self.qvalue_eps/4
        return policy_matrix

    def get_distance_to_slow_policy(self):
        # The slow policy always picks the second action (index 1)
        slow_policy_matrix = torch.zeros((self.n_states, 4))
        slow_policy_matrix[:, 1] = 1

        # The distance is the sum of the absolute differences between the two policy matrices
        policy_matrix = self.get_policy_matrix()
        return torch.sum(torch.abs(policy_matrix - slow_policy_matrix)).item()

    def get_distance_to_fast_policy(self):
        # The fast policy always picks the last action (index 3)
        fast_policy_matrix = torch.zeros((self.n_states, 4))
        fast_policy_matrix[:, 3] = 1

        # The distance is the sum of the absolute differences between the two policy matrices
        policy_matrix = self.get_policy_matrix()
        return torch.sum(torch.abs(policy_matrix - fast_policy_matrix)).item()

    #override
    def get_agent_details(self, agent_detail_args):
        n_states = agent_detail_args["n_states"]
        self.n_states = n_states # Used in distance methods
        value_lr = agent_detail_args["value_lr"]
        target_eps = agent_detail_args["target_eps"]
        value_max_grad = agent_detail_args["value_max_grad"]
        qvalue_eps = agent_detail_args["qvalue_eps"]
        self.qvalue_eps = qvalue_eps # Used in distance methods

        class QValueNet(nn.Module):
            def __init__(self, n_states):
                super().__init__()
                self.n_states = n_states
                self.net = nn.Linear(n_states, 4)

            def forward(self, x):
                # Transform input to onehot, converting to torch.float32
                x = torch.nn.functional.one_hot(x, num_classes=self.n_states).float().squeeze(-2)
                return self.net(x)
        qvalue_network = QValueNet(n_states).to(self.device)
        eval_policy_module = QValueActor(qvalue_network, in_keys=["observation"], spec=self.env.action_spec).to(self.device)
        loss_module = DQNLoss(value_network=eval_policy_module, action_space="one-hot", double_dqn=True)
        optim = torch.optim.Adam(loss_module.parameters(), lr=value_lr)

        self.target_net = SoftUpdate(loss_module, eps=target_eps)

        max_grad_dict = {
            "value": (loss_module.value_network_params.parameters, value_max_grad)
        }

        train_policy_module = TensorDictSequential(
            eval_policy_module,
            EGreedyModule(spec=self.env.action_spec, eps_init=qvalue_eps, eps_end=qvalue_eps)
        )

        return loss_module, ["loss"], [optim], max_grad_dict, train_policy_module, eval_policy_module

    #override
    def update_callback(self):
        self.target_net.step()

    #override
    def train_info_dict_callback(self, td):
        return {
            "state distribution": wandb.Histogram(td["observation"].cpu()),
            "action distribution": wandb.Histogram(td["action"].argmax(dim=-1).cpu()),
            "qvalues": math.sqrt(sum((p**2).sum() for p in self.loss_module.value_network_params.parameters())),
            "distance to slow policy": self.get_distance_to_slow_policy(),
            "distance to fast policy": self.get_distance_to_fast_policy()
        }

    #override
    def eval_info_dict_callback(self, td):
        eval_normal_return = calc_return((td["next", "normal_reward"]).flatten(), self.env.gamma)
        eval_true_return = calc_return((td["next", "normal_reward"]+td["next","constraint_reward"]).flatten(), self.env.gamma)
        return {
            "eval normal return": eval_normal_return,
            "eval true return": eval_true_return,
        }


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
