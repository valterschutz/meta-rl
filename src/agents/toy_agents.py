import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

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

# from torchrl.modules import IndependentNormal

# from utils import OneHotLayer, print_computational_graph


import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from networks import MetaPolicyNet, MetaValueNet


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


# TODO: in progress
class DDPGToyAgent:
    def __init__(
        self,
        n_states,
        n_actions,
        action_spec,
        num_optim_epochs,
        buffer_size,
        sub_batch_size,
        device,
        max_grad_norm,
        policy_lr,
        qvalue_lr,
        gamma,
        target_eps,
        mode="train",
    ):
        super().__init__()

        self.n_states = n_states
        self.n_actions = n_actions
        self.action_spec = action_spec

        self.buffer_size = buffer_size
        self.num_optim_epochs = num_optim_epochs
        self.sub_batch_size = sub_batch_size
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.policy_lr = policy_lr
        self.qvalue_lr = qvalue_lr
        self.gamma = gamma
        self.target_eps = target_eps

        self.replay_buffer = TensorDictReplayBuffer(
            batch_size=sub_batch_size,
            storage=LazyTensorStorage(max_size=buffer_size, device=self.device),
        )

        self.reset(
            mode=mode,
        )

    def policy(self, td):
        return self.policy_module(td)

    def reset(
        self,
        mode: str,
    ):
        # Policy
        actor_net = nn.Sequential(
            # nn.Linear(self.n_states, actor_hidden_units),
            # nn.ReLU(),
            nn.Linear(self.n_states, self.n_actions),
            # nn.Tanh(),
        ).to(self.device)
        self.policy_module = Actor(
            spec=self.action_spec,
            module=actor_net,
            in_keys=["state"],
            out_keys=["action"],
        )

        # Critic
        class QValueNet(nn.Module):
            def __init__(self, n_states, n_actions):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(n_states + n_actions, 1),
                )

            def forward(self, state, action):
                x = torch.cat((state, action), dim=-1)
                return self.net(x)

        self.qvalue_net = QValueNet(self.n_states, self.n_actions).to(self.device)
        self.qvalue_module = ValueOperator(
            self.qvalue_net,
            in_keys=["state", "action"],
            out_keys=["state_action_value"],
        )

        if mode == "train":
            self.policy_module.train()
            self.qvalue_module.train()
        elif mode == "eval":
            self.policy_module.eval()
            self.qvalue_module.eval()
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.loss_module = DDPGLoss(
            actor_network=self.policy_module,
            value_network=self.qvalue_module,
        )
        self.loss_module.make_value_estimator(gamma=self.gamma)
        self.target_updater = SoftUpdate(self.loss_module, eps=self.target_eps)

        self.policy_optim = torch.optim.Adam(
            self.policy_module.parameters(), lr=self.policy_lr
        )
        self.qvalue_optim = torch.optim.Adam(
            self.qvalue_module.parameters(), lr=self.policy_lr
        )
        self.replay_buffer.empty()

        self.use_constraints = False

    def process_batch(self, td, verbose=False):
        self.replay_buffer.extend(td.clone().detach())  # Detach before extending
        max_grad_norm = 0
        losses_actor = []
        losses_value = []
        for i in range(self.num_optim_epochs):
            # for i in range(1):
            sub_base_td = self.replay_buffer.sample(self.sub_batch_size)
            sub_base_td = td
            if self.use_constraints:
                sub_base_td["next", "reward"] = (
                    sub_base_td["next", "normal_reward"]
                    + sub_base_td["next", "constraint_reward"]
                )
            else:
                sub_base_td["next", "reward"] = sub_base_td["next", "normal_reward"]

            loss_td = self.loss_module(sub_base_td)
            loss = loss_td["loss_actor"] + loss_td["loss_value"]
            losses_actor.append(loss_td["loss_actor"].mean().item())
            losses_value.append(loss_td["loss_value"].mean().item())
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                self.loss_module.parameters(), self.max_grad_norm
            )
            max_grad_norm = max(grad_norm.item(), max_grad_norm)

            self.policy_optim.step()
            self.policy_optim.zero_grad()
            self.qvalue_optim.step()
            self.qvalue_optim.zero_grad()

            self.target_updater.step()
        losses = TensorDict(
            {
                "loss_actor": torch.tensor(
                    sum(losses_actor) / len(losses_actor),
                    device=self.device,
                    dtype=torch.float32,
                ),
                "loss_value": torch.tensor(
                    sum(losses_value) / len(losses_value),
                    device=self.device,
                    dtype=torch.float32,
                ),
            },
            batch_size=(),
        )

        return losses, max_grad_norm


class SACToyAgent:
    def __init__(
        self,
        n_states,
        # n_actions,
        action_spec,
        num_optim_epochs,
        buffer_size,
        sub_batch_size,
        device,
        max_grad_norm,
        lr,
        gamma,
        target_eps,
        target_entropy,
        min_buffer_size,
        mode="train",
    ):
        super().__init__()

        self.n_states = n_states
        # self.n_actions = n_actions
        self.action_spec = action_spec
        # self.n_actions = action_spec.n

        self.buffer_size = buffer_size
        self.min_buffer_size = min_buffer_size
        self.num_optim_epochs = num_optim_epochs
        self.sub_batch_size = sub_batch_size
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.lr = lr
        self.gamma = gamma
        self.target_eps = target_eps
        self.target_entropy = target_entropy

        self.replay_buffer = TensorDictReplayBuffer(
            batch_size=sub_batch_size,
            storage=LazyTensorStorage(max_size=buffer_size, device=self.device),
            sampler=SamplerWithoutReplacement(),
        )
        # TODO: PRS
        # self.replay_buffer = ReplayBuffer(
        #     storage=LazyTensorStorage(max_size=self.buffer_size, device=self.device),
        #     sampler=SamplerWithoutReplacement(),
        # )

        self.reset(
            mode=mode,
        )

    def policy(self, td):
        return self.policy_module(td)

    def reset(
        self,
        mode: str,
    ):
        n_actions = self.action_spec.n

        # Policy
        actor_net = nn.Sequential(
            nn.Linear(self.n_states, n_actions),
        ).to(self.device)
        policy_module = TensorDictModule(
            actor_net, in_keys=["state"], out_keys=["logits"]
        )
        self.policy_module = ProbabilisticActor(
            policy_module,
            spec=self.action_spec,
            in_keys=["logits"],
            # out_keys=["action"],
            distribution_class=OneHotCategorical,
            default_interaction_type=InteractionType.RANDOM,
        )

        self.qvalue_net = nn.Sequential(nn.Linear(self.n_states, n_actions)).to(
            self.device
        )
        self.qvalue_module = ValueOperator(
            self.qvalue_net,
            in_keys=["state"],
            out_keys=["action_value"],
        )

        if mode == "train":
            self.policy_module.train()
            self.qvalue_module.train()
        elif mode == "eval":
            self.policy_module.eval()
            self.qvalue_module.eval()
        else:
            raise ValueError(f"Unknown mode: {mode}")

        use_target_entropy = type(self.target_entropy) is float
        print(f"{use_target_entropy=}")
        self.loss_module = DiscreteSACLoss(
            actor_network=self.policy_module,
            qvalue_network=self.qvalue_module,
            action_space=self.action_spec,
            num_actions=n_actions,
            fixed_alpha=use_target_entropy,
            target_entropy=(self.target_entropy if use_target_entropy else "auto"),
            loss_function="l2",
            # delay_qvalue=True,
        )
        self.loss_module.make_value_estimator(ValueEstimators.TD0, gamma=self.gamma)
        self.target_updater = SoftUpdate(self.loss_module, eps=self.target_eps)

        self.policy_optim = torch.optim.Adam(
            self.loss_module.actor_network_params.parameters(), lr=self.lr
        )
        self.qvalue_optim = torch.optim.Adam(
            self.loss_module.qvalue_network_params.parameters(), lr=self.lr
        )
        self.replay_buffer.empty()

        self.use_constraints = False

    def process_batch(self, td):
        self.replay_buffer.extend(td.clone().detach())  # Detach before extending
        if len(self.replay_buffer) < self.min_buffer_size:
            return TensorDict(), {}
        mean_policy_grad_norm = []
        mean_qvalue_grad_norm = []
        losses_actor = []
        losses_qvalue = []
        losses_alpha = []
        mean_entropy = []
        for i in range(self.num_optim_epochs):
            sub_base_td = self.replay_buffer.sample(self.sub_batch_size)
            if self.use_constraints:
                sub_base_td["next", "reward"] = (
                    sub_base_td["next", "normal_reward"]
                    + sub_base_td["next", "constraint_reward"]
                )
            else:
                sub_base_td["next", "reward"] = sub_base_td["next", "normal_reward"]
            self.policy_optim.zero_grad()
            self.qvalue_optim.zero_grad()
            loss_td = self.loss_module(sub_base_td)
            loss = (
                loss_td["loss_actor"] + loss_td["loss_qvalue"] + loss_td["loss_alpha"]
            )
            losses_actor.append(loss_td["loss_actor"].mean().item())
            losses_qvalue.append(loss_td["loss_qvalue"].mean().item())
            losses_alpha.append(loss_td["loss_alpha"].mean().item())
            mean_entropy.append(loss_td["entropy"].mean().item())
            loss.backward()

            qvalue_grad_norm = nn.utils.clip_grad_norm_(
                self.loss_module.qvalue_network_params.parameters(), self.max_grad_norm
            )
            mean_qvalue_grad_norm.append(qvalue_grad_norm.item())
            policy_grad_norm = nn.utils.clip_grad_norm_(
                self.loss_module.actor_network_params.parameters(), self.max_grad_norm
            )
            mean_policy_grad_norm.append(policy_grad_norm.item())

            self.policy_optim.step()
            self.qvalue_optim.step()

            self.target_updater.step()
        losses = TensorDict(
            {
                "loss_actor": torch.tensor(
                    sum(losses_actor) / len(losses_actor),
                    device=self.device,
                    dtype=torch.float32,
                ),
                "loss_qvalue": torch.tensor(
                    sum(losses_qvalue) / len(losses_qvalue),
                    device=self.device,
                    dtype=torch.float32,
                ),
                "loss_alpha": torch.tensor(
                    sum(losses_alpha) / len(losses_alpha),
                    device=self.device,
                    dtype=torch.float32,
                ),
            },
            batch_size=(),
        )
        additional_info = {
            "entropy": sum(mean_entropy) / len(mean_entropy),
            "mean_policy_grad_norm": sum(mean_policy_grad_norm)
            / len(mean_policy_grad_norm),
            "mean_qvalue_grad_norm": sum(mean_qvalue_grad_norm)
            / len(mean_qvalue_grad_norm),
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
