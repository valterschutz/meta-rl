import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List

from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import TargetNetUpdater

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from tensordict import TensorDict
from tensordict.nn import InteractionType, TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.data import (
    Binary,
    Categorical,
    ReplayBuffer,
    TensorDictReplayBuffer,
    UnboundedContinuous,
)
from torchrl.data.replay_buffers import (
    LazyMemmapStorage,
    LazyTensorStorage,
    PrioritizedSampler,
    SamplerWithoutReplacement,
)
from torchrl.modules import OneHotCategorical, TanhNormal, TruncatedNormal
from torchrl.modules.tensordict_module import (
    Actor,
    ProbabilisticActor,
    SafeModule,
    ValueOperator,
)
from torchrl.objectives import (
    TD3Loss,
    A2CLoss,
    ClipPPOLoss,
    DDPGLoss,
    DiscreteSACLoss,
    SACLoss,
    SoftUpdate,
    ValueEstimators,
)
from torchrl.objectives.value import GAE, TD0Estimator

import wandb

import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


class PPOMetaAgent:
    def __init__(
        self,
        state_spec,
        action_spec,
        num_optim_epochs,
        buffer_size,
        device,
        max_policy_grad_norm,
        max_value_grad_norm,
        lr,
        gamma,
        hidden_units,
        clip_epsilon,
        use_entropy,
        entropy_coef,
        critic_coef,
        policy_module_state_dict=None,
        value_module_state_dict=None,
        mode="train",
    ):
        super().__init__()

        self.state_spec = state_spec
        self.action_spec = action_spec

        self.num_optim_epochs = num_optim_epochs
        self.buffer_size = buffer_size
        self.device = device
        self.max_policy_grad_norm = max_policy_grad_norm
        self.max_value_grad_norm = max_value_grad_norm
        self.lr = lr
        self.gamma = gamma
        self.hidden_units = hidden_units
        self.clip_epsilon = clip_epsilon
        self.use_entropy = use_entropy
        self.entropy_coef = entropy_coef
        self.critic_coef = critic_coef

        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=buffer_size, device=self.device),
            sampler=SamplerWithoutReplacement(),
        )

        self.reset(
            mode=mode,
            policy_module_state_dict=policy_module_state_dict,
            value_module_state_dict=value_module_state_dict,
        )

    def reset(
        self,
        mode: str,
        policy_module_state_dict=None,
        value_module_state_dict=None,
    ):
        # state_keys = ["base_true_mean_reward", "last_action"]  # Must haves
        state_keys = ["last_action"]  # Sanity check?
        # state_keys = ["step"]
        n_states = len(state_keys)
        # state_keys = ["step"]

        # Policy
        policy_net = MetaPolicyNet(n_states, self.hidden_units, self.device)
        policy_module = TensorDictModule(
            policy_net, in_keys=state_keys, out_keys=["loc", "scale"]
        )
        self.policy_module = ProbabilisticActor(
            module=policy_module,
            spec=self.action_spec,
            in_keys=["loc", "scale"],
            out_keys=["action"],
            distribution_class=TruncatedNormal,
            distribution_kwargs={"low": 0.0, "high": 1.0},
            default_interaction_type=InteractionType.RANDOM,
            return_log_prob=True,
        )
        if policy_module_state_dict is not None:
            self.policy_module.load_state_dict(policy_module_state_dict)

        # Value
        value_net = MetaValueNet(n_states, self.hidden_units, self.device)
        self.value_module = ValueOperator(
            value_net,
            in_keys=state_keys,
            out_keys=["state_value"],
        )
        if value_module_state_dict is not None:
            self.value_module.load_state_dict(value_module_state_dict)

        if mode == "train":
            self.policy_module.train()
            self.value_module.train()
        elif mode == "eval":
            self.policy_module.eval()
            self.value_module.eval()
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.loss_module = ClipPPOLoss(
            actor_network=self.policy_module,
            critic_network=self.value_module,
            clip_epsilon=self.clip_epsilon,
            entropy_bonus=self.use_entropy,
            entropy_coef=self.entropy_coef,
            critic_coef=self.critic_coef,
            loss_critic_type="smooth_l1",
        )
        self.advantage_module = TD0Estimator(
            gamma=self.gamma, value_network=self.value_module
        )
        self.optim = torch.optim.Adam(self.loss_module.parameters(), lr=self.lr)

        self.replay_buffer.empty()

    def policy(self, td):
        return self.policy_module(td)

    def process_batch(self, td, verbose=False):
        # Process a single batch of data and return losses and maximum grad norm
        td["sample_log_prob"] = td["sample_log_prob"].detach()
        td["action"] = td["action"].detach()
        avg_policy_grad_norm = 0
        avg_value_grad_norm = 0
        losses_objective = []
        losses_critic = []
        losses_entropy = []
        for i in range(self.num_optim_epochs):
            self.advantage_module(td)

            sub_base_td = td.clone()

            loss_td = self.loss_module(sub_base_td)

            # Loss propagation
            loss = (
                loss_td["loss_objective"] + loss_td["loss_critic"] + loss_td["entropy"]
            )
            loss.backward()

            # Log losses
            losses_objective.append(loss_td["loss_objective"].mean().item())
            losses_critic.append(loss_td["loss_critic"].mean().item())
            losses_entropy.append(loss_td["loss_entropy"].mean().item())

            # Policy gradient norm
            policy_grad_norm = nn.utils.clip_grad_norm_(
                self.policy_module.parameters(), self.max_policy_grad_norm
            )
            avg_policy_grad_norm += (1 / (i + 1)) * (
                policy_grad_norm.item() - avg_policy_grad_norm
            )

            # Value gradient norm
            value_grad_norm = nn.utils.clip_grad_norm_(
                self.value_module.parameters(), self.max_value_grad_norm
            )
            avg_value_grad_norm += (1 / (i + 1)) * (
                value_grad_norm.item() - avg_value_grad_norm
            )

            self.optim.step()
            self.optim.zero_grad()

        losses = TensorDict(
            {
                "loss_objective": torch.tensor(
                    sum(losses_objective) / len(losses_objective),
                    device=self.device,
                    dtype=torch.float32,
                ),
                "loss_critic": torch.tensor(
                    sum(losses_critic) / len(losses_critic),
                    device=self.device,
                    dtype=torch.float32,
                ),
                "loss_entropy": torch.tensor(
                    sum(losses_entropy) / len(losses_entropy),
                    device=self.device,
                    dtype=torch.float32,
                ),
            },
            batch_size=(),
        )

        return losses, avg_policy_grad_norm, avg_value_grad_norm


class OffpolicyAgent:
    optims: List[
        torch.optim.Optimizer
    ]  # List of optimizers to call `zero_grad` and `step` on during training
    loss_keys: List[
        str
    ]  # List of keys that will be read from the loss_td tensordict after evaluating loss
    loss_module: (
        LossModule  # Should have `.make_value_estimator` method already called on it
    )
    num_optim_epochs: int  # How many times to sample from the replay buffer and optimize the loss for each environment step
    buffer_size: int
    min_buffer_size: (
        int  # Minimum number of samples in the replay buffer before training starts
    )
    sub_base_size: int  # Number of samples to draw from the replay buffer for each optimization step
    max_grad_norm: float
    target_updater: TargetNetUpdater
    device: torch.device  # Where to store the replay buffer

    def __init__(
        self,
        num_optim_epochs,
        buffer_size,
        sub_batch_size,
        device,
        max_grad_norm,
        min_buffer_size,
        target_updater,
        optims,
        loss_keys,
        loss_module,
        **kwargs,
    ):
        super().__init__()

        self.buffer_size = buffer_size
        self.min_buffer_size = min_buffer_size
        self.num_optim_epochs = num_optim_epochs
        self.sub_batch_size = sub_batch_size
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.optims = optims
        self.loss_keys = loss_keys
        self.loss_module = loss_module
        self.target_updater = target_updater
        self.device = device

        self.replay_buffer = TensorDictReplayBuffer(
            batch_size=sub_batch_size,
            storage=LazyTensorStorage(max_size=buffer_size, device=self.device),
            sampler=SamplerWithoutReplacement(),
        )

        # This will change as training progresses
        self.use_constraints = False

    def process_batch(self, td):
        """
        Process a single batch of data, adding it to the replay buffer and training the agent if the buffer is large enough.
        Returns losses as specified by `self.loss_keys` and also gradient norm.
        """
        self.replay_buffer.extend(td.clone().detach())  # Detach before extending
        if len(self.replay_buffer) < self.min_buffer_size:
            return TensorDict(), {}
        loss_metrics = {key: [] for key in self.loss_keys}
        mean_grad_norm = []
        for _ in range(self.num_optim_epochs):
            sub_base_td = self.replay_buffer.sample(self.sub_batch_size)
            if self.use_constraints:
                sub_base_td["next", "reward"] = (
                    sub_base_td["next", "normal_reward"]
                    + sub_base_td["next", "constraint_reward"]
                )
            else:
                sub_base_td["next", "reward"] = sub_base_td["next", "normal_reward"]
            loss_td = self.loss_module(sub_base_td)
            loss = sum(v for k, v in loss_td.items() if k in self.loss_keys)
            # Save loss metrics
            for k in self.loss_keys:
                loss_metrics[k].append(loss_td[k].mean().item())
            loss.backward()

            grad_norm = nn.utils.clip_grad_norm_(
                self.loss_module.parameters(), self.max_grad_norm
            )
            mean_grad_norm.append(grad_norm.item())

            for optim in self.optims:
                optim.step()
                optim.zero_grad()

            self.target_updater.step()
        losses = TensorDict(
            {
                k: torch.tensor(
                    sum(v) / len(v),
                    device=self.device,
                    dtype=torch.float32,
                )
                for k, v in loss_metrics.items()
            },
            batch_size=(),
        )
        additional_info = {
            "mean_grad_norm": sum(mean_grad_norm) / len(mean_grad_norm),
            "mean normal reward": np.mean(td["next", "normal_reward"].cpu().numpy()),
            "mean constraint reward": np.mean(
                td["next", "constraint_reward"].cpu().numpy()
            ),
        }

        return losses, additional_info


class ValueIterationToyAgent:
    def __init__(self, env, gamma):
        self.env = env
        self.tol = 1e-3
        self.Q = torch.zeros((env.n_states, env.action_spec.n), device=env.device)
        # self.lr = 1e-1
        self.gamma = gamma

    def update_values(self):
        # Update values
        n_states = self.env.n_states
        n_actions = self.env.action_spec.n
        td = TensorDict(
            {
                "state": F.one_hot(
                    torch.arange(n_states - 1, device=self.env.device),
                    num_classes=n_states,
                ).repeat_interleave(n_actions, dim=0),
                "action": F.one_hot(
                    torch.arange(n_actions, device=self.env.device),
                    num_classes=n_actions,
                ).repeat(n_states - 1, 1),
                "step_count": torch.zeros(
                    (n_states - 1) * n_actions, device=self.env.device
                ),
            },
            batch_size=[(n_states - 1) * n_actions],
        )
        td = self.env.step(td)
        delta = float("inf")
        while delta > self.tol:
            delta = 0.0
            for state in range(n_states - 1):
                for action in range(n_actions):
                    idx = state * n_actions + action
                    next_state = td["next", "state"][idx].argmax(dim=-1).item()
                    reward = td["next", "reward"][idx].item()
                    prev_Q = self.Q[state, action].item()
                    self.Q[state, action] = (
                        reward + self.gamma * self.Q[next_state, :].max().item()
                    )
                    delta = max(
                        delta,
                        (prev_Q - self.Q[state, action]).abs().item(),
                    )
            # fail

    def policy(self, td):
        action_idx = self.Q[td["state"].argmax(dim=-1)].argmax(dim=-1)
        td["action"] = F.one_hot(action_idx, num_classes=self.env.action_spec.n).to(
            torch.float32
        )
        return td


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


def get_toy_agent(agent_type, agent_config, env):
    if agent_type == "SAC":
        return SACToyAgent(
            n_states=env.n_states,
            action_spec=env.action_spec,
            num_optim_epochs=agent_config["num_optim_epochs"],
            buffer_size=agent_config["buffer_size"],
            sub_batch_size=agent_config["sub_batch_size"],
            device=agent_config["device"],
            max_grad_norm=agent_config["max_grad_norm"],
            lr=agent_config["lr"],
            gamma=agent_config["gamma"],
            target_eps=agent_config["target_eps"],
            target_entropy=agent_config["target_entropy"],
            min_buffer_size=agent_config["min_buffer_size"],
            mode="train",
        )
    elif agent_type == "DDPG":
        return DDPGToyAgent(
            n_states=env.n_states,
            action_spec=env.action_spec,
            num_optim_epochs=agent_config["num_optim_epochs"],
            buffer_size=agent_config["buffer_size"],
            sub_batch_size=agent_config["sub_batch_size"],
            device=agent_config["device"],
            max_grad_norm=agent_config["max_grad_norm"],
            policy_lr=agent_config["policy_lr"],
            qvalue_lr=agent_config["qvalue_lr"],
            gamma=agent_config["gamma"],
            target_eps=agent_config["target_eps"],
            mode="train",
        )
    elif agent_type == "TD3":
        return TD3ToyAgent(
            n_states=env.n_states,
            action_spec=env.action_spec,
            num_optim_epochs=agent_config["num_optim_epochs"],
            buffer_size=agent_config["buffer_size"],
            sub_batch_size=agent_config["sub_batch_size"],
            device=agent_config["device"],
            max_grad_norm=agent_config["max_grad_norm"],
            lr=agent_config["lr"],
            gamma=agent_config["gamma"],
            target_eps=agent_config["target_eps"],
            min_buffer_size=agent_config["min_buffer_size"],
            mode="train",
        )
