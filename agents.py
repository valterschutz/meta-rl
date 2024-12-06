import torch
import tensordict
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC, abstractmethod
from typing import Optional

from tensordict.nn import InteractionType


from torchrl.modules.tensordict_module import ProbabilisticActor, ValueOperator
from torchrl.data.replay_buffers import LazyTensorStorage, SamplerWithoutReplacement
from torchrl.data import Categorical, Binary, UnboundedContinuous, ReplayBuffer
from torchrl.objectives import ClipPPOLoss, A2CLoss, DDPGLoss
from torchrl.objectives.value import GAE

# from torchrl.modules import IndependentNormal

from utils import OneHotLayer, print_computational_graph

from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.modules import TruncatedNormal

import wandb


class PPOAgent(ABC):
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
        clip_epsilon,
        use_entropy,
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

        self.clip_epsilon = clip_epsilon
        self.use_entropy = use_entropy

        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=self.buffer_size, device=self.device),
            sampler=SamplerWithoutReplacement(),
        )

        # Variables expected to be set by subclasses
        # self.policy
        # self.value_module
        # self.advantage_module

        self.reset(
            mode=mode,
            policy_module_state_dict=policy_module_state_dict,
            value_module_state_dict=value_module_state_dict,
        )

    @abstractmethod
    def initialize_policy(self, mode: str):
        """Expected to return `self.policy_module`."""

    @abstractmethod
    def initialize_critic(self, mode: str):
        """Expected to return `self.value_module` and `self.advantage_module`."""

    @abstractmethod
    def policy(self, td):
        """Expected to fill the action key of the input tensor dictionary."""

    def reset(
        self, mode: str, policy_module_state_dict=None, value_module_state_dict=None
    ):
        self.policy_module = self.initialize_policy(mode=mode)
        if policy_module_state_dict is not None:
            self.policy_module.load_state_dict(policy_module_state_dict)
        self.value_module = self.initialize_critic(mode=mode)
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
        self.advantage_module = GAE(
            gamma=self.gamma,
            lmbda=self.lmbda,
            value_network=self.value_module,
        )
        self.loss_module = ClipPPOLoss(
            actor_network=self.policy_module,
            critic_network=self.value_module,
            clip_epsilon=self.clip_epsilon,
            entropy_bonus=self.use_entropy,
        )
        self.optim = torch.optim.Adam(self.loss_module.parameters(), lr=self.lr)
        self.replay_buffer.empty()

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
                loss = (
                    loss_td["loss_objective"]
                    + loss_td["loss_critic"]
                    + (0 if not self.use_entropy else loss_td["loss_entropy"])
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


class BaseAgent(PPOAgent):
    def __init__(
        self,
        **kwargs,
    ):
        # We expect state_spec and action_spec to both be catogorical
        super().__init__(**kwargs)

    def initialize_policy(self, mode: str):
        n_states = self.state_spec["state"].n
        n_actions = self.action_spec.n

        actor_net = nn.Sequential(
            OneHotLayer(num_classes=n_states),
            nn.Linear(n_states, n_actions),
        ).to(self.device)
        policy_module = TensorDictModule(
            actor_net, in_keys=["state"], out_keys=["logits"]
        )
        policy_module = ProbabilisticActor(
            module=policy_module,
            spec=self.action_spec,
            in_keys=["logits"],
            distribution_class=torch.distributions.Categorical,
            default_interaction_type=InteractionType.RANDOM,
            return_log_prob=True,
        )

        return policy_module

    def initialize_critic(self, mode: str):
        n_states = self.state_spec["state"].n

        value_net = nn.Sequential(
            OneHotLayer(num_classes=n_states),
            nn.Linear(n_states, 1),
        ).to(self.device)
        value_module = ValueOperator(
            value_net, in_keys=["state"], out_keys=["state_value"]
        )

        return value_module

    def policy(self, td):
        return self.policy_module(td)


class MetaPolicyNet(nn.Module):
    def __init__(self, hidden_units, n_states, n_outputs, device):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, n_outputs),
            nn.Sigmoid(),
            # nn.ReLU(),  # If using Beta distribution
            # NormalParamExtractor(),
        ).to(device)

    def forward(self, *args):
        # def forward(self, base_mean_reward, base_std_reward, last_action, step):
        # x = torch.cat(
        #     (base_mean_reward, base_std_reward, last_action, step), dim=-1
        # ).clone()
        x = torch.cat(args, dim=-1)
        return self.net(x)


class MetaValueNet(nn.Module):
    def __init__(self, hidden_units, n_states, device):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, 1),
        ).to(device)

    def forward(self, *args):
        x = torch.cat(args, dim=-1)
        return self.net(x)

    # def forward(self, x):
    #     return self.net(x)


class MetaAgent(PPOAgent):
    def __init__(self, hidden_units, **kwargs):
        self.hidden_units = hidden_units
        super().__init__(**kwargs)

    def initialize_policy(self, mode: str):
        # policy_net = MetaPolicyNet(
        #     hidden_units=self.hidden_units, n_states=4, n_outputs=1, device=self.device
        # ).to(self.device)
        policy_net = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid(),
        )
        policy_module = TensorDictModule(
            policy_net,
            # in_keys=["base_mean_reward", "base_std_reward", "last_action", "step"],
            in_keys=["step"],
            # out_keys=["logits"],
            out_keys=["loc"],
        )
        policy_module = ProbabilisticActor(
            module=policy_module,
            spec=self.action_spec,
            # in_keys=["logits"],
            in_keys=["loc"],
            return_log_prob=True,
            default_interaction_type=InteractionType.MODE,
            distribution_class=TruncatedNormal,
            distribution_kwargs={"scale": torch.tensor([0.1])},
            # distribution_class=torch.distributions.bernoulli.Bernoulli,
            # distribution_class=torch.distributions.normal.Normal,
        )

        return policy_module

    def initialize_critic(self, mode: str):
        value_net = MetaValueNet(
            hidden_units=self.hidden_units, n_states=4, device=self.device
        ).to(self.device)
        value_module = ValueOperator(
            value_net,
            in_keys=["base_mean_reward", "base_std_reward", "last_action", "step"],
            out_keys=["state_value"],
        )

        return value_module

    def policy(self, td):
        # td = td.clone()  # Ensure no in-place modifications
        return self.policy_module(td)


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
    td["action"] = torch.tensor(1, device=td.device)
    return td


def fast_policy(td):
    # Always go up in even states, otherwise go right
    if td["state"] % 2 == 0:
        td["action"] = torch.tensor(3, device=td.device)
    else:
        td["action"] = torch.tensor(1, device=td.device)
    return td
