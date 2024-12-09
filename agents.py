import torch
import tensordict
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC, abstractmethod
from typing import Optional

from tensordict.nn import InteractionType


from torchrl.modules.tensordict_module import (
    ProbabilisticActor,
    ValueOperator,
    ActorCriticOperator,
    SafeModule,
)
from torchrl.data.replay_buffers import LazyTensorStorage, SamplerWithoutReplacement
from torchrl.data import Categorical, Binary, UnboundedContinuous, ReplayBuffer
from torchrl.objectives import (
    ClipPPOLoss,
    A2CLoss,
    DDPGLoss,
    SACLoss,
    DiscreteSACLoss,
    ValueEstimators,
    SoftUpdate,
)
from torchrl.objectives.value import GAE

# from torchrl.modules import IndependentNormal

from utils import OneHotLayer, print_computational_graph

from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.modules import TruncatedNormal, OneHotCategorical

import wandb

from networks import MetaPolicyNet, MetaQValueNet, MetaValueNet


class MetaAgent:
    def __init__(
        self,
        state_spec,
        action_spec,
        num_optim_epochs,
        buffer_size,
        sub_batch_size,
        device,
        max_grad_norm,
        lr,
        gamma,
        lmbda,
        hidden_units,
        policy_module_state_dict=None,
        value_module_state_dict=None,
        mode="train",
    ):

        self.state_spec = state_spec
        self.action_spec = action_spec

        self.buffer_size = buffer_size
        self.num_optim_epochs = num_optim_epochs
        self.sub_batch_size = sub_batch_size
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.lr = lr
        self.gamma = gamma
        self.lmbda = lmbda

        self.hidden_units = hidden_units

        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=self.buffer_size, device=self.device),
            sampler=SamplerWithoutReplacement(),
        )

        self.reset(
            mode=mode,
            policy_module_state_dict=policy_module_state_dict,
            value_module_state_dict=value_module_state_dict,
        )

    def initialize_policy(self):
        policy_net = MetaPolicyNet(
            hidden_units=self.hidden_units, n_states=4, n_outputs=1, device=self.device
        ).to(self.device)
        policy_module = TensorDictModule(
            policy_net,
            in_keys=["base_mean_reward", "base_std_reward", "last_action", "step"],
            out_keys=["loc"],
        )
        policy_module = ProbabilisticActor(
            module=policy_module,
            spec=self.action_spec,
            in_keys=["loc"],
            return_log_prob=True,
            default_interaction_type=InteractionType.MODE,
            distribution_class=TruncatedNormal,
            distribution_kwargs={"scale": torch.tensor([0.1])},
        )

        return policy_module

    def initialize_qvalue(self):
        n_actions = self.action_spec.n
        qvalue_net = MetaQValueNet(
            hidden_units=self.hidden_units,
            n_states=4,
            n_actions=n_actions,
            device=self.device,
        ).to(self.device)
        qvalue_module = ValueOperator(
            qvalue_net,
            in_keys=[
                "base_mean_reward",
                "base_std_reward",
                "last_action",
                "step",
                "action",
            ],
            out_keys=["state_action_value"],
        )

        return qvalue_module

    def initialize_value(self):
        value_net = MetaValueNet(
            hidden_units=self.hidden_units, n_states=4, device=self.device
        ).to(self.device)
        value_module = ValueOperator(
            value_net,
            in_keys=["base_mean_reward", "base_std_reward", "last_action", "step"],
            out_keys=["state_value"],
        )

        return value_module

    def reset(
        self,
        mode: str,
        policy_module_state_dict=None,
        qvalue_module_state_dict=None,
        value_module_state_dict=None,
    ):
        self.policy_module = self.initialize_policy()
        if policy_module_state_dict is not None:
            self.policy_module.load_state_dict(policy_module_state_dict)
        self.qvalue_module = self.initialize_qvalue()
        if qvalue_module_state_dict is not None:
            self.qvalue_module.load_state_dict(qvalue_module_state_dict)
        self.value_module = self.initialize_value()
        if value_module_state_dict is not None:
            self.value_module.load_state_dict(value_module_state_dict)
        if mode == "train":
            self.policy_module.train()
            self.qvalue_module.train()
            self.value_module.train()
        elif mode == "eval":
            self.policy_module.eval()
            self.qvalue_module.eval()
            self.value_module.eval()
        else:
            raise ValueError(f"Unknown mode: {mode}")
        self.advantage_module = GAE(
            gamma=self.gamma,
            lmbda=self.lmbda,
            value_network=self.value_module,
        )
        self.loss_module = SACLoss(
            actor_network=self.policy_module,
            qvalue_network=self.qvalue_module,
            value_network=self.value_module,
        )
        self.optim = torch.optim.Adam(self.loss_module.parameters(), lr=self.lr)
        self.replay_buffer.empty()

    def policy(self, td):
        return self.policy_module(td)

    def process_batch(self, td, verbose=False):
        # Detach sample_log_prob and action from the graph. TODO: understand why
        td["sample_log_prob"] = td["sample_log_prob"].detach()
        td["action"] = td["action"].detach()

        # Process a single batch of data and return losses and maximum grad norm
        times_to_sample = len(td) // self.sub_batch_size
        max_grad_norm = 0
        losses_objective = []
        losses_critic = []
        losses_entropy = []
        for i in range(self.num_optim_epochs):
            self.advantage_module(td)
            self.replay_buffer.extend(td.clone().detach())  # Detach before extending
            for j in range(times_to_sample):
                sub_base_td = self.replay_buffer.sample(self.sub_batch_size)

                self.optim.zero_grad()
                loss_td = self.loss_module(sub_base_td)
                fail  # Check which losses are available
                loss = (
                    loss_td["loss_objective"]
                    + loss_td["loss_critic"]
                    # + (0 if not self.use_entropy else loss_td["loss_entropy"])
                )
                losses_objective.append(loss_td["loss_objective"].mean().item())
                losses_critic.append(loss_td["loss_critic"].mean().item())
                losses_entropy.append(
                    0 if not self.use_entropy else loss_td["loss_entropy"].mean().item()
                )
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    self.loss_module.parameters(), self.max_grad_norm
                )
                max_grad_norm = max(grad_norm.item(), max_grad_norm)
                self.optim.step()
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

        return losses, max_grad_norm


class BaseAgent:
    def __init__(
        self,
        state_spec,
        action_spec,
        num_optim_epochs,
        buffer_size,
        sub_batch_size,
        device,
        max_grad_norm,
        lr,
        gamma,
        hidden_units,
        target_eps,
        target_entropy,
        actor_critic_module_state_dict=None,
        mode="train",
    ):

        self.state_spec = state_spec
        self.action_spec = action_spec

        self.buffer_size = buffer_size
        self.num_optim_epochs = num_optim_epochs
        self.sub_batch_size = sub_batch_size
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.lr = lr
        self.gamma = gamma
        self.hidden_units = hidden_units
        self.target_eps = target_eps
        self.target_entropy = target_entropy

        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=self.buffer_size, device=self.device),
            sampler=SamplerWithoutReplacement(),
        )
        # TODO: try prioritized sampling

        self.reset(
            mode=mode,
            actor_critic_module_state_dict=actor_critic_module_state_dict,
        )

    def reset(
        self,
        mode: str,
        actor_critic_module_state_dict=None,
    ):
        n_states = self.state_spec["state"].n
        n_actions = self.action_spec.n

        # Common hidden network for both policy and critic
        hidden_net = nn.Sequential(
            OneHotLayer(num_classes=n_states),
            nn.Linear(n_states, self.hidden_units),
            nn.ReLU(),
            nn.Linear(self.hidden_units, self.hidden_units),
        ).to(self.device)
        # hidden_module = SafeModule(hidden_net, in_keys=["state"], out_keys=["hidden"])

        # Policy head
        actor_net = nn.Sequential(
            hidden_net,
            nn.Linear(self.hidden_units, n_actions),
        ).to(self.device)
        policy_module = TensorDictModule(
            actor_net, in_keys=["state"], out_keys=["logits"]
        )
        self.policy_module = ProbabilisticActor(
            module=policy_module,
            spec=self.action_spec,
            in_keys=["logits"],
            # distribution_class=torch.distributions.Categorical,
            distribution_class=OneHotCategorical,
            default_interaction_type=InteractionType.RANDOM,
            return_log_prob=True,
        )

        # Action value head
        qvalue_net = nn.Sequential(
            hidden_net,
            nn.Linear(self.hidden_units, n_actions),
        ).to(self.device)
        self.qvalue_module = ValueOperator(
            qvalue_net, in_keys=["state"], out_keys=["action_value"]
        )

        # State value head, necessary for GAE
        value_net = nn.Sequential(
            hidden_net,
            nn.Linear(self.hidden_units, 1),
        ).to(self.device)
        self.value_module = ValueOperator(
            value_net, in_keys=["state"], out_keys=["state_value"]
        )

        # TODO: load state dict
        # if actor_critic_module_state_dict is not None:
        #     self.combined_module.load_state_dict(actor_critic_module_state_dict)

        # self.value_module = self.actor_critic_module.get_value_operator()

        if mode == "train":
            # self.value_module.train()
            self.policy_module.train()
            self.qvalue_module.train()
        elif mode == "eval":
            # self.value_module.eval()
            self.policy_module.eval()
            self.qvalue_module.eval()
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # self.advantage_module = GAE(
        #     gamma=self.gamma,
        #     lmbda=self.lmbda,
        #     value_network=self.value_module,
        # )

        self.loss_module = DiscreteSACLoss(
            actor_network=self.policy_module,
            qvalue_network=self.qvalue_module,
            action_space=self.action_spec,
            num_actions=self.action_spec.n,
            target_entropy=self.target_entropy,
        )
        self.loss_module.make_value_estimator(
            ValueEstimators.TD0, gamma=self.gamma
        )  # Que?
        self.optim = torch.optim.Adam(self.loss_module.parameters(), lr=self.lr)
        self.target_updater = SoftUpdate(self.loss_module, eps=self.target_eps)
        self.replay_buffer.empty()

    def policy(self, td):
        return self.policy_module(td)

    def process_batch(self, td, verbose=False):
        # Detach sample_log_prob and action from the graph. TODO: understand why
        td["sample_log_prob"] = td["sample_log_prob"].detach()
        td["action"] = td["action"].detach()

        # Process a single batch of data and return losses and maximum grad norm
        # times_to_sample = len(td) // self.sub_batch_size
        max_grad_norm = 0
        losses_alpha = []
        losses_actor = []
        losses_qvalue = []
        self.replay_buffer.extend(td.clone().detach())  # Detach before extending
        for i in range(self.num_optim_epochs):
            # self.value_estimator(td)
            sub_base_td = self.replay_buffer.sample(self.sub_batch_size)
            # self.advantage_module(sub_base_td)

            self.optim.zero_grad()
            loss_td = self.loss_module(sub_base_td)
            loss = (
                loss_td["loss_alpha"] + loss_td["loss_actor"] + loss_td["loss_qvalue"]
            )
            losses_alpha.append(loss_td["loss_alpha"].mean().item())
            losses_actor.append(loss_td["loss_actor"].mean().item())
            losses_qvalue.append(loss_td["loss_qvalue"].mean().item())
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                self.loss_module.parameters(), self.max_grad_norm
            )
            max_grad_norm = max(grad_norm.item(), max_grad_norm)
            self.optim.step()
            self.target_updater.step()
        losses = TensorDict(
            {
                "loss_alpha": torch.tensor(
                    sum(losses_alpha) / len(losses_alpha),
                    device=self.device,
                    dtype=torch.float32,
                ),
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
            },
            batch_size=(),
        )

        return losses, max_grad_norm


class ValueIterationAgent:
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
                "state": torch.arange(n_states - 1, device=self.env.device)[:, None]
                .repeat(1, n_actions)
                .reshape(-1),
                "action": torch.arange(n_actions, device=self.env.device).repeat(
                    n_states - 1
                ),
                "step_count": torch.zeros(
                    (n_states - 1) * n_actions, device=self.env.device
                ),
            },
            batch_size=[(n_states - 1) * n_actions],
        )
        td = self.env.step(td)
        # print(f"state: {td['state']}")
        # print(f"action: {td['action']}")
        # print(f"next_state: {td['next','state']}")
        # print(f"reward: {td['next','reward']}")
        delta = float("inf")
        while delta > self.tol:
            delta = 0.0
            for state in range(n_states - 1):
                for action in range(n_actions):
                    idx = state * n_actions + action
                    next_state = td["next", "state"][idx].item()
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
        td["action"] = self.Q[td["state"].long()].argmax(dim=-1)
        return td


def slow_policy(td):
    # Always go right
    td["action"] = torch.tensor([0, 1, 0, 0], device=td.device)
    return td


def fast_policy(td):
    # Always go up in even states, otherwise go right
    if td["state"] % 2 == 0:
        td["action"] = torch.tensor([0, 0, 0, 1], device=td.device)
    else:
        td["action"] = torch.tensor([0, 1, 0, 0], device=td.device)
    return td
