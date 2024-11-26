import torch
import torch.nn as nn
import torch.nn.functional as F

from torchrl.modules.tensordict_module import ProbabilisticActor, ValueOperator
from torchrl.data.replay_buffers import LazyTensorStorage, SamplerWithoutReplacement
from torchrl.data import Categorical, Binary, UnboundedContinuous, ReplayBuffer
from torchrl.objectives import ClipPPOLoss, A2CLoss
from torchrl.objectives.value import GAE

from utils import OneHotLayer

from tensordict import TensorDict
from tensordict.nn import TensorDictModule


class DiscreteACAgent:
    def __init__(self, n_states, n_actions, device, w_lr, theta_lr, num_optim_epochs):
        self.n_states = n_states
        self.n_actions = n_actions

        self.device = device
        self.w_lr = w_lr
        self.theta_lr = theta_lr
        self.num_optim_epochs = num_optim_epochs

        self.reset()

    def reset(self):
        self.w = torch.randn(
            self.n_states, device=self.device
        )  # weights for value function
        self.theta = torch.randn(
            self.n_states * self.n_actions, device=self.device
        )  # weights for policy

    def get_state_feature(self, state):
        return F.one_hot(state.to(torch.long), num_classes=self.n_states).to(
            torch.float32
        )

    def get_state_action_feature(self, state, action):
        return F.one_hot(
            (state * self.n_actions + action).to(torch.long),
            num_classes=self.n_states * self.n_actions,
        ).to(torch.float32)

    def get_state_action_features(self, state):
        # Get all state-action features for a given state, one feature per column
        r = torch.zeros((self.n_states * self.n_actions, self.n_actions))
        for i in range(self.n_actions):
            r[:, i] = self.get_state_action_feature(state, i)
        return r

    def process_batch(self, td):
        td_errors = []
        for j in range(self.num_optim_epochs):
            for i in range(len(td)):
                state = td["state"][i]
                action = td["action"][i]
                reward = td["next", "reward"][i]
                next_state = td["next", "state"][i]
                # print(f"{state=}, {action=}, {reward=}, {next_state=}")
                td_error = reward + self.w[next_state] - self.w[state]
                self.w += self.w_lr * td_error * self.get_state_feature(state)
                action_probs = self.get_action_probabilities(state)
                grad_log_pi = (
                    self.get_state_action_feature(state, action)
                    - self.get_state_action_features(state) @ action_probs
                )
                self.theta += self.theta_lr * td_error * grad_log_pi
                if j == self.num_optim_epochs - 1:
                    td_errors.append(td_error)
        return td_errors

    def get_action_preferences(self, state):
        return self.theta[state * self.n_actions : (state + 1) * self.n_actions]

    def get_action_probabilities(self, state):
        return F.softmax(self.get_action_preferences(state), dim=-1)

    def exploration_policy(self, td):
        state = td["state"].item()
        action_preferences = self.get_action_preferences(state)
        # sample according to preferences (logits)
        td["action"] = torch.distributions.Categorical(
            logits=action_preferences
        ).sample()
        return td

    def explotation_policy(self, td):
        # Pick the action with the highest preference
        action_preferences = self.get_action_preferences(td["state"])
        td["action"] = torch.argmax(action_preferences).to(torch.int32)
        return td


class MetaAgent:
    def __init__(self, state_spec, action_spec, device, max_grad_norm, lr):
        # We expect state_spec to be UnboundedContinuous and action_spec to be Binary
        assert isinstance(state_spec["state"], UnboundedContinuous)
        assert isinstance(action_spec, Binary)
        self.state_spec = state_spec
        self.action_spec = action_spec
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.lr = lr
        self.hidden_units = 16

        self.reset()

    def reset(self):
        self.initialize_policy()
        self.initialize_critic()
        self.loss_module = ClipPPOLoss(
            actor_network=self.policy,
            critic_network=self.value_module,
            clip_epsilon=0.2,
            entropy_bonus=False,
        )
        self.optim = torch.optim.Adam(self.loss_module.parameters(), lr=self.lr)

    def initialize_policy(self):
        self.actor_net = nn.Sequential(
            nn.Linear(self.state_spec["state"].shape[-1], self.hidden_units),
            nn.Tanh(),
            nn.Linear(self.hidden_units, 1),
            nn.Sigmoid(),  # Interpret action as probability to set constraint
        ).to(self.device)
        policy = TensorDictModule(self.actor_net, in_keys=["state"], out_keys=["probs"])
        self.policy = ProbabilisticActor(
            module=policy,
            spec=self.action_spec,
            in_keys=["probs"],
            distribution_class=torch.distributions.Bernoulli,
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
            gamma=0.98,
            lmbda=0.96,
            value_network=self.value_module,
        )

    def process_batch(self, td):
        # Since td will only contain a single sample, we don't bother with several updates
        self.advantage_module(td)
        # Detach sample_log_prob??? TODO
        td["sample_log_prob"] = td["sample_log_prob"].detach()

        self.optim.zero_grad()
        loss_td = self.loss_module(td)
        loss = loss_td["loss_objective"] + loss_td["loss_critic"]
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            self.loss_module.parameters(), self.max_grad_norm
        )
        self.optim.step()

    # def act(self, td):
    #     # Update actions which will affect next base agent batch
    #     td = self.policy(td)
    #     # Apply meta action
    #     action_prob = td["probs"]
    #     action = torch.bernoulli(action_prob).bool().item()
    #     td["action"] = action
    #     return td
