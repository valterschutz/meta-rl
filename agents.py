import torch
import tensordict
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC, abstractmethod

from torchrl.modules.tensordict_module import ProbabilisticActor, ValueOperator
from torchrl.data.replay_buffers import LazyTensorStorage, SamplerWithoutReplacement
from torchrl.data import Categorical, Binary, UnboundedContinuous, ReplayBuffer
from torchrl.objectives import ClipPPOLoss, A2CLoss, DDPGLoss
from torchrl.objectives.value import GAE
from torchrl.modules import IndependentNormal

from utils import OneHotLayer

from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor


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

        self.reset()

    @abstractmethod
    def initialize_policy(self):
        pass

    @abstractmethod
    def initialize_critic(self):
        pass

    def reset(self):
        self.initialize_policy()
        self.initialize_critic()
        self.loss_module = ClipPPOLoss(
            actor_network=self.policy,
            critic_network=self.value_module,
            clip_epsilon=self.clip_epsilon,
            entropy_bonus=self.use_entropy,
        )
        self.optim = torch.optim.Adam(self.loss_module.parameters(), lr=self.lr)
        self.replay_buffer.empty()

    def process_batch(self, td):
        # Process a single batch of data and return losses and maximum grad norm
        times_to_sample = len(td) // self.sub_batch_size
        max_grad_norm = 0
        losses_objective = []
        losses_critic = []
        losses_entropy = []
        for i in range(self.num_optim_epochs):
            self.advantage_module(td)
            self.replay_buffer.extend(td)
            for j in range(times_to_sample):
                sub_base_td = self.replay_buffer.sample(self.sub_batch_size)

                self.optim.zero_grad()
                loss_td = self.loss_module(sub_base_td)
                loss = (
                    loss_td["loss_objective"]
                    + loss_td["loss_critic"]
                    + (0 if not self.use_entropy else loss_td["loss_entropy"])
                )
                # print(f"{loss.requires_grad=}")
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

    def initialize_policy(self):
        n_states = self.state_spec["state"].n
        n_actions = self.action_spec.n

        self.actor_net = nn.Sequential(
            OneHotLayer(num_classes=n_states),
            nn.Linear(n_states, n_actions),
        ).to(self.device)
        self.policy = TensorDictModule(
            self.actor_net, in_keys=["state"], out_keys=["logits"]
        )
        self.policy = ProbabilisticActor(
            module=self.policy,
            spec=self.action_spec,
            in_keys=["logits"],
            distribution_class=torch.distributions.Categorical,
            return_log_prob=True,
        )

    def initialize_critic(self):
        n_states = self.state_spec["state"].n

        self.value_net = nn.Sequential(
            OneHotLayer(num_classes=n_states),
            # nn.Linear(n_states, self.hidden_units),
            # nn.Tanh(),
            nn.Linear(n_states, 1),
        ).to(self.device)
        self.value_module = ValueOperator(
            self.value_net, in_keys=["state"], out_keys=["state_value"]
        )
        self.advantage_module = GAE(
            gamma=self.gamma,
            lmbda=self.lmbda,
            value_network=self.value_module,
        )


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
        # Policy is deterministic
        td["action"] = self.Q[td["state"].long(), :].argmax(dim=-1)
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


class MetaAgent(PPOAgent):
    def __init__(self, hidden_units, **kwargs):
        super().__init__(**kwargs)
        self.hidden_units = hidden_units

    def initialize_policy(self):
        self.actor_net = nn.Sequential(
            nn.Linear(self.state_spec["state"].shape[-1], self.hidden_units),
            nn.Tanh(),
            nn.Linear(self.hidden_units, 2),
            NormalParamExtractor(),
        ).to(self.device)
        policy = TensorDictModule(
            self.actor_net, in_keys=["state"], out_keys=["loc", "scale"]
        )
        self.policy = ProbabilisticActor(
            module=policy,
            spec=self.action_spec,
            in_keys=["loc", "scale"],
            distribution_class=IndependentNormal,
            # default_interaction_type="InteractionType.MODE",
            return_log_prob=True,
        )

    def initialize_critic(self):
        self.value_net = nn.Sequential(
            nn.Linear(self.state_spec["state"].shape[-1], self.hidden_units),
            nn.Tanh(),
            nn.Linear(self.hidden_units, 1),
        ).to(self.device)
        self.value_module = ValueOperator(
            self.value_net, in_keys=["state"], out_keys=["state_value"]
        )
        self.advantage_module = GAE(
            gamma=self.gamma,
            lmbda=self.lmbda,
            value_network=self.value_module,
        )
