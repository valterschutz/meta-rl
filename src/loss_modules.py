import torch
import torch.nn as nn
from tensordict.nn import InteractionType, TensorDictModule
from torch.distributions import OneHotCategorical
from torchrl.modules import (NormalParamExtractor, ProbabilisticActor,
                             TruncatedNormal, ValueOperator)
from torchrl.objectives import DiscreteSACLoss, SACLoss, ValueEstimators, TD3Loss, ClipPPOLoss
from torchrl.modules.tensordict_module import Actor

def get_discrete_ppo_loss_module(
    n_states, action_spec, gamma
):
    n_actions = action_spec.n
    hidden_units = 20
    # Policy
    actor_net = nn.Sequential(
        nn.Linear(n_states, n_actions),
        nn.ReLU(),
        nn.Linear(hidden_units, hidden_units),
        nn.ReLU(),
        nn.Linear(hidden_units, n_actions),
    )
    policy_module = TensorDictModule(actor_net, in_keys=["state"], out_keys=["logits"])
    policy_module = ProbabilisticActor(
        policy_module,
        spec=action_spec,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=OneHotCategorical,
        default_interaction_type=InteractionType.RANDOM,
        return_log_prob=True,
    )

    value_net = nn.Sequential(
        nn.Linear(n_states, hidden_units),
        nn.ReLU(),
        nn.Linear(hidden_units, hidden_units),
        nn.ReLU(),
        nn.Linear(hidden_units, 1),
    )
    value_module = ValueOperator(
        value_net,
        in_keys=["state"],
        out_keys=["state_value"],
    )
    loss_module = ClipPPOLoss(
        actor_network=policy_module,
        critic_network=value_module,
    )
    loss_module.make_value_estimator(ValueEstimators.TD0, gamma=gamma)
    return loss_module

def get_continuous_ppo_loss_module(
    n_states, n_actions, action_spec, gamma, action_low, action_high
):
    hidden_units = 256
    # Policy
    actor_net = nn.Sequential(
        nn.Linear(n_states, hidden_units),
        nn.ReLU(),
        nn.Linear(hidden_units, hidden_units),
        nn.ReLU(),
        nn.Linear(hidden_units, n_actions * 2),
        NormalParamExtractor()
    )
    policy_module = TensorDictModule(
        actor_net, in_keys=["state"], out_keys=["loc", "scale"]
    )
    if action_low is None:
        action_low = action_spec.low.item()
    if action_high is None:
        action_high = action_spec.high.item()
    policy_module = ProbabilisticActor(
        policy_module,
        spec=action_spec,
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TruncatedNormal,
        distribution_kwargs={"low": action_low, "high": action_high},
        default_interaction_type=InteractionType.RANDOM,
        return_log_prob=True
    )

    value_net = nn.Sequential(
        nn.Linear(n_states, hidden_units),
        nn.ReLU(),
        nn.Linear(hidden_units, hidden_units),
        nn.ReLU(),
        nn.Linear(hidden_units, 1),
    )
    value_module = ValueOperator(
        value_net,
        in_keys=["state"],
        out_keys=["state_value"],
    )
    loss_module = ClipPPOLoss(
        actor_network=policy_module,
        critic_network=value_module,
    )
    loss_module.make_value_estimator(ValueEstimators.TD0, gamma=gamma)
    return loss_module

def get_discrete_sac_loss_module(
    n_states, action_spec, gamma
):
    n_actions = action_spec.n
    hidden_units = 20
    # Policy
    actor_net = nn.Sequential(
        nn.Linear(n_states, hidden_units),
        nn.ReLU(),
        nn.Linear(hidden_units, hidden_units),
        nn.ReLU(),
        nn.Linear(hidden_units, n_actions),
    )
    policy_module = TensorDictModule(actor_net, in_keys=["state"], out_keys=["logits"])
    policy_module = ProbabilisticActor(
        policy_module,
        spec=action_spec,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=OneHotCategorical,
        default_interaction_type=InteractionType.RANDOM,
    )

    qvalue_net = nn.Sequential(
        nn.Linear(n_states, hidden_units),
        nn.ReLU(),
        nn.Linear(hidden_units, hidden_units),
        nn.ReLU(),
        nn.Linear(hidden_units, n_actions),
    )
    qvalue_module = ValueOperator(
        qvalue_net,
        in_keys=["state"],
        out_keys=["action_value"],
    )
    loss_module = DiscreteSACLoss(
        actor_network=policy_module,
        qvalue_network=qvalue_module,
        action_space=action_spec,
        num_actions=n_actions,
        # delay_qvalue=True, # TODO: check this
    )
    loss_module.make_value_estimator(ValueEstimators.TD0, gamma=gamma)
    return loss_module


def get_continuous_sac_loss_module(
    n_states, n_actions, action_spec, gamma, action_low=None, action_high=None
):
    hidden_units = 256
    # Policy
    actor_net = nn.Sequential(
        nn.Linear(n_states, hidden_units),
        nn.ReLU(),
        nn.Linear(hidden_units, hidden_units),
        nn.ReLU(),
        nn.Linear(hidden_units, n_actions * 2),
        NormalParamExtractor()
    )
    policy_module = TensorDictModule(
        actor_net, in_keys=["state"], out_keys=["loc", "scale"]
    )
    if action_low is None:
        action_low = action_spec.low.item()
    if action_high is None:
        action_high = action_spec.high.item()
    policy_module = ProbabilisticActor(
        policy_module,
        spec=action_spec,
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TruncatedNormal,
        distribution_kwargs={"low": action_low, "high": action_high},
        default_interaction_type=InteractionType.RANDOM,
    )

    class QValueNet(nn.Module):
        def __init__(self, n_states, n_actions):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_states + n_actions, hidden_units),
                nn.ReLU(),
                nn.Linear(hidden_units, hidden_units),
                nn.ReLU(),
                nn.Linear(hidden_units, 1),
            )

        def forward(self, state, action):
            return self.net(torch.cat([state, action], dim=-1))
    qvalue_net = QValueNet(n_states=n_states, n_actions=n_actions)
    qvalue_module = ValueOperator(
        qvalue_net,
        in_keys=["state", "action"],
        out_keys=["state_action_value"],
    )
    loss_module = SACLoss(
        actor_network=policy_module,
        qvalue_network=qvalue_module,
    )
    loss_module.make_value_estimator(ValueEstimators.TD0, gamma=gamma)
    return loss_module

def get_continuous_td3_loss_module(
    n_states, n_actions, action_spec, gamma
):
    hidden_units = 256
    # Policy
    actor_net = nn.Sequential(
        nn.Linear(n_states, hidden_units),
        nn.Tanh(),
        nn.Linear(hidden_units, hidden_units),
        nn.Tanh(),
        nn.Linear(hidden_units, n_actions),
        nn.Tanh(),
    )
    policy_module = Actor(
        actor_net, in_keys=["state"], out_keys=["action"], spec=action_spec
    )

    class QValueNet(nn.Module):
        def __init__(self, n_states, n_actions):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_states + n_actions, hidden_units),
                nn.ReLU(),
                nn.Linear(hidden_units, hidden_units),
                nn.ReLU(),
                nn.Linear(hidden_units, 1),
            )

        def forward(self, state, action):
            return self.net(torch.cat([state, action], dim=-1))
    qvalue_net = QValueNet(n_states=n_states, n_actions=n_actions)
    qvalue_module = ValueOperator(
        qvalue_net,
        in_keys=["state", "action"],
        out_keys=["state_action_value"],
    )
    # use_target_entropy = isinstance(target_entropy, (int, float))
    loss_module = TD3Loss(
        actor_network=policy_module,
        qvalue_network=qvalue_module,
        action_spec=action_spec,
    )
    loss_module.make_value_estimator(ValueEstimators.TD0, gamma=gamma)
    return loss_module
