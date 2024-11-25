import torch
import torch.nn as nn

from torchrl.modules.tensordict_module import ProbabilisticActor, ValueOperator
from torchrl.data.replay_buffers import LazyTensorStorage, SamplerWithoutReplacement
from torchrl.data import Categorical, Binary, UnboundedContinuous, ReplayBuffer
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from utils import OneHotLayer

from tensordict import TensorDictModule


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
    ):
        # We expect state_spec and action_spec to both be catogorical
        assert isinstance(state_spec["pos"], Categorical)
        assert isinstance(action_spec, Categorical)
        n_states = state_spec["pos"].numel()
        n_actions = action_spec.numel()
        # Base agent networks
        self.hidden_units = 4
        self.actor_net = nn.Sequential(
            OneHotLayer(num_classes=n_states),
            nn.Linear(n_states, self.hidden_units),
            nn.Tanh(),
            nn.Linear(self.hidden_units, n_actions),
        ).to(device)
        self.policy = TensorDictModule(
            self.actor_net, in_keys=["pos"], out_keys=["logits"]
        )
        self.policy = ProbabilisticActor(
            module=self.policy,
            spec=action_spec,
            in_keys=["logits"],
            distribution_class=Categorical,
            return_log_prob=True,
        )
        self.value_net = nn.Sequential(
            OneHotLayer(num_classes=n_states),
            nn.Linear(n_states, self.hidden_units),
            nn.Tanh(),
            nn.Linear(self.hidden_units, 1),
        ).to(device)
        self.value_module = ValueOperator(
            self.value_net, in_keys=["pos"], out_keys=["state_value"]
        )
        self.loss_module = ClipPPOLoss(
            actor_network=self.policy,
            critic_network=self.value_module,
            clip_epsilon=0.2,
            entropy_bonus=False,
        )
        self.advantage_module = GAE(
            gamma=0.98,
            lmbda=0.96,
            value_network=self.value_module,
        )
        self.optim = torch.optim.Adam(self.loss_module.parameters(), lr=lr)

        self.buffer_size = buffer_size

        self.num_optim_epochs = num_optim_epochs

        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=buffer_size, device=device),
            sampler=SamplerWithoutReplacement(),
        )

        self.sub_batch_size = sub_batch_size
        self.max_grad_norm = max_grad_norm
        self.lr = lr

    def process_batch(self, base_td):
        # Process a single batch of data
        for _ in range(self.num_optim_epochs):
            self.advantage_module(base_td)
            self.replay_buffer.extend(base_td)
            for _ in range(len(base_td) // self.sub_batch_size):
                sub_base_td = self.replay_buffer.sample(self.sub_batch_size)
                loss = self.loss_module(sub_base_td)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

    def reset(self):
        self.replay_buffer.empty()
        # Reset the base agent networks
        self.policy.reset()
        self.value_module.reset()


class MetaAgent:
    def __init__(self, state_spec, action_spec, device, max_grad_norm, lr):
        # We expect state_spec to be UnboundedContinuous and action_spec to be Binary
        assert isinstance(state_spec["state"], UnboundedContinuous)
        assert isinstance(action_spec, Binary)
        self.hidden_units = 16
        self.actor_net = nn.Sequential(
            nn.Linear(state_spec.ndim, self.hidden_units),
            nn.Tanh(),
            nn.Linear(self.hidden_units, 1),
            nn.Sigmoid(),  # Interpret action as probability to set constraint
        ).to(device)
        policy = TensorDictModule(self.actor_net, in_keys=["state"], out_keys=["probs"])
        self.policy = ProbabilisticActor(
            module=policy,
            spec=action_spec,
            in_keys=["probs"],
            distribution_class=torch.distributions.Bernoulli,
            return_log_prob=True,
        )
        self.value_net = nn.Sequential(
            nn.Linear(2, self.hidden_units),
            nn.Tanh(),
            nn.Linear(self.hidden_units, 1),
        ).to(device)
        self.value_module = ValueOperator(
            self.value_net, in_keys=["state"], out_keys=["state_value"]
        )
        self.loss_module = ClipPPOLoss(
            actor_network=policy,
            critic_network=self.value_module,
            clip_epsilon=0.2,
            entropy_bonus=False,
        )
        self.advantage_module = GAE(
            gamma=0.98,
            lmbda=0.96,
            value_network=self.value_module,
        )
        self.optim = torch.optim.Adam(self.loss_module.parameters(), lr=lr)

        self.max_grad_norm = max_grad_norm
        self.lr = lr

    def process_batch(self, td):
        # Since td will only contain a single sample, we don't bother with several updates
        self.advantage_module(td)
        loss_vals = self.loss_module(td)
        loss = loss_vals["loss_objective"] + loss_vals["loss_critic"]
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            self.loss_module.parameters(), self.max_grad_norm
        )
        self.optim.step()
        self.optim.zero_grad()

    # def act(self, td):
    #     # Update actions which will affect next base agent batch
    #     td = self.policy(td)
    #     # Apply meta action
    #     action_prob = td["probs"]
    #     action = torch.bernoulli(action_prob).bool().item()
    #     td["action"] = action
    #     return td
