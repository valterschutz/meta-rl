import pickle
from pathlib import Path
import yaml
import torch.nn as nn
import torch

from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers import LazyTensorStorage
from tensordict.nn.distributions import NormalParamExtractor
from tensordict.nn.probabilistic import InteractionType
from tensordict.nn import TensorDictModule
from tensordict import TensorDict
from torchrl.modules.tensordict_module import ProbabilisticActor, ValueOperator, Actor
from torchrl.objectives import SoftUpdate, SACLoss, DDPGLoss
from torchrl.modules import TanhNormal


class SACCartpoleAgent:
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
        lr,
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
        self.lr = lr
        self.gamma = gamma
        self.target_eps = target_eps

        self.replay_buffer = TensorDictReplayBuffer(
            batch_size=sub_batch_size,
            # storage=LazyMemmapStorage(buffer_size, device=self.device),
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
        actor_hidden_units = 256
        actor_net = nn.Sequential(
            nn.Linear(self.n_states, actor_hidden_units),
            nn.ReLU(),
            nn.Linear(actor_hidden_units, actor_hidden_units),
            nn.ReLU(),
            nn.Linear(actor_hidden_units, 2 * self.n_actions),
            NormalParamExtractor(),
        ).to(self.device)
        policy_module = TensorDictModule(
            actor_net, in_keys=["state"], out_keys=["loc", "scale"]
        )
        self.policy_module = ProbabilisticActor(
            module=policy_module,
            spec=self.action_spec,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            default_interaction_type=InteractionType.RANDOM,
            return_log_prob=True,
        )

        # Critic
        critic_hidden_units = 256

        class QValueNet(nn.Module):
            def __init__(self, n_states, n_actions, hidden_units):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(n_states + n_actions, hidden_units),
                    nn.ReLU(),
                    nn.Linear(hidden_units, hidden_units),
                    nn.ReLU(),
                    nn.Linear(hidden_units, 1),
                )

            def forward(self, state, action):
                x = torch.cat((state, action), dim=-1)
                return self.net(x)

        self.qvalue_net = QValueNet(
            self.n_states, self.n_actions, critic_hidden_units
        ).to(self.device)
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

        self.loss_module = SACLoss(
            actor_network=self.policy_module,
            qvalue_network=self.qvalue_module,
        )
        self.loss_module.make_value_estimator(gamma=self.gamma)
        self.target_updater = SoftUpdate(self.loss_module, eps=self.target_eps)

        self.optim = torch.optim.Adam(self.loss_module.parameters(), lr=self.lr)
        self.replay_buffer.empty()

        self.use_constraints = False

    def process_batch(self, td, verbose=False):
        self.replay_buffer.extend(td.clone().detach())  # Detach before extending
        max_grad_norm = 0
        losses_actor = []
        losses_qvalue = []
        losses_alpha = []
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
            loss = (
                loss_td["loss_actor"] + loss_td["loss_qvalue"] + loss_td["loss_alpha"]
            )
            losses_actor.append(loss_td["loss_actor"].mean().item())
            losses_qvalue.append(loss_td["loss_qvalue"].mean().item())
            losses_alpha.append(loss_td["loss_alpha"].mean().item())
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                self.loss_module.parameters(), self.max_grad_norm
            )
            max_grad_norm = max(grad_norm.item(), max_grad_norm)

            self.optim.step()
            self.optim.zero_grad()

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

        return losses, max_grad_norm


class DDPGCartpoleAgent:
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
            # storage=LazyMemmapStorage(buffer_size, device=self.device),
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
            nn.Linear(self.n_states, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, self.n_actions),
            nn.Tanh(),
        ).to(self.device)
        self.policy_module = Actor(
            spec=self.action_spec,
            module=actor_net,
            in_keys=["state"],
            out_keys=["action"],
        )

        class QValueNet(nn.Module):
            def __init__(self, n_states, n_actions):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(n_states + n_actions, 400),
                    nn.ReLU(),
                    nn.Linear(400, 300),
                    nn.ReLU(),
                    nn.Linear(300, 1),
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
            self.qvalue_module.parameters(), lr=self.qvalue_lr
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

    def save_policy(self, path: Path):
        # Pickle the policy module
        torch.save(self.policy_module, path)

    def save_qvalues(self, path: Path):
        # Pickle the qvalue module
        torch.save(self.qvalue_module, path)

    @staticmethod
    def load_policy(path: Path):
        # Load agent from a pickle
        policy_module = torch.load(path)
        return policy_module


def get_cartpole_agent(algo_type, action_spec):
    if algo_type == "SAC":
        with open("configs/cartpole/sac.yaml", encoding="UTF-8") as f:
            config = yaml.safe_load(f)
        return SACCartpoleAgent(
            n_states=5,
            n_actions=1,
            action_spec=action_spec,
            num_optim_epochs=config["num_optim_epochs"],
            buffer_size=config["buffer_size"],
            sub_batch_size=config["sub_batch_size"],
            device=config["device"],
            max_grad_norm=config["max_grad_norm"],
            lr=config["lr"],
            gamma=config["gamma"],
            target_eps=config["target_eps"],
            mode="train",
        )
    elif algo_type == "DDPG":
        with open("configs/agents/cartpole/cartpole_ddpg.yaml", encoding="UTF-8") as f:
            config = yaml.safe_load(f)
        return (
            DDPGCartpoleAgent(
                n_states=5,
                n_actions=1,
                action_spec=action_spec,
                num_optim_epochs=config["num_optim_epochs"],
                buffer_size=config["buffer_size"],
                sub_batch_size=config["sub_batch_size"],
                device=config["device"],
                max_grad_norm=config["max_grad_norm"],
                policy_lr=config["policy_lr"],
                qvalue_lr=config["qvalue_lr"],
                gamma=config["gamma"],
                target_eps=config["target_eps"],
                mode="train",
            ),
            config,
        )
