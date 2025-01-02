import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from tensordict.nn.probabilistic import InteractionType
from torch import multiprocessing, nn
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (Compose, DMControlEnv, DoubleToFloat,
                          ObservationNorm, StepCounter, TransformedEnv,
                          set_gym_backend)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.transforms import CatTensors
from torchrl.envs import ParallelEnv
from torchrl.envs.utils import (ExplorationType, check_env_specs,
                                set_exploration_type)
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import DiscreteSACLoss, SoftUpdate
from torchrl.objectives.value import GAE
from torchrl.record import VideoRecorder
from torchrl.record.loggers.csv import CSVLogger
from torchrl.modules import OneHotCategorical
from tqdm import tqdm
from torchrl.objectives import ValueEstimators
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers import PrioritizedSampler, ListStorage

import wandb

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "./"))

from envs.toy_env import ToyEnv
from utils import calc_return

class BaseAgent:
    def __init__(self, action_spec, state_spec, device, buffer_size, min_buffer_size, batch_size, sub_batch_size, num_epochs, lr, gamma, target_eps, alpha, beta, max_grad_norm):
        self.action_spec = action_spec
        self.state_spec = state_spec
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

        n_states = self.state_spec["state"].shape[-1]
        n_actions = self.action_spec.shape[-1]
        actor_net = nn.Sequential(
            nn.Linear(n_states, n_actions, device=device),
        )

        policy_module = TensorDictModule(
            actor_net, in_keys=["state"], out_keys=["logits"]
        )

        self.policy_module = ProbabilisticActor(
            module=policy_module,
            spec=self.action_spec,
            in_keys=["logits"],
            distribution_class=OneHotCategorical,
            default_interaction_type=InteractionType.RANDOM, # TODO: should this be random?
            return_log_prob=True,
        )

        qvalue_net = nn.Sequential(
            nn.Linear(n_states, n_actions, device=device),
        )
        self.qvalue_module = ValueOperator(
            module=qvalue_net,
            in_keys=["state"],
            out_keys=["action_value"],
        )

        self.loss_module = DiscreteSACLoss(
            actor_network=self.policy_module,
            qvalue_network=self.qvalue_module,
            num_actions=n_actions,
            target_entropy=0.0, # TODO: does this work?
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

def train_base_agent(device, total_frames, min_buffer_size, n_states, return_x, return_y, percentage_constraints_active, times_to_eval, log, progress_bar):
    env_max_steps = 100
    big_reward = 10.0
    gamma = 0.99
    batch_size = 200
    sub_batch_size = 20
    num_epochs = 100
    lr = 1e-2
    target_eps = 0.99
    alpha = 0.7
    beta = 0.5
    max_grad_norm = 100.0

    n_batches = total_frames // batch_size
    eval_every_n_batch = n_batches // times_to_eval

    transforms = Compose(
        StepCounter(max_steps=env_max_steps),
    )
    x, y = ToyEnv.calculate_xy(n_states=n_states, return_x=return_x, return_y=return_y, big_reward=big_reward, gamma=gamma)
    env = ToyEnv(
        left_reward=x,
        right_reward=x,
        down_reward=y,
        up_reward=y,
        n_states=n_states,
        big_reward=big_reward,
        constraints_active=False,
        random_start=False,
        seed=None,
        device=device
    )
    env = TransformedEnv(
        env,
        transforms
    )
    # check_env_specs(env)

    agent = BaseAgent(
        state_spec=env.state_spec,
        action_spec=env.action_spec,
        device=device,
        buffer_size = total_frames,
        min_buffer_size = min_buffer_size,
        batch_size = batch_size,
        sub_batch_size = sub_batch_size,
        num_epochs = num_epochs,
        lr = lr,
        gamma = gamma,
        target_eps = target_eps,
        alpha=alpha,
        beta=beta,
        max_grad_norm = max_grad_norm
    )

    rand_collector = SyncDataCollector(
        env,
        None,
        frames_per_batch=agent.batch_size,
        total_frames=agent.min_buffer_size,
        split_trajs=False,
        device=device,
    )

    collector = SyncDataCollector(
        env,
        agent.policy_module,
        frames_per_batch=agent.batch_size,
        total_frames=total_frames-agent.min_buffer_size,
        split_trajs=False,
        device=device,
    )

    if progress_bar:
        pbar = tqdm(total=total_frames)
    batch_to_activate_constraints = int(n_batches * percentage_constraints_active)
    for td in rand_collector:
        agent.process_batch(td, constraints_active=False)
        if progress_bar:
            pbar.update(td.numel())

    eval_returns = []
    try:
        for i, td in enumerate(collector):
            td["action"] = td["action"].to(torch.float32) # Due to bug in torchrl, need to manually cast to float
            collector.update_policy_weights_() # Check if this is necessary

            loss_dict, info_dict = agent.process_batch(td, constraints_active=i >= batch_to_activate_constraints)

            if log:
                wandb.log({
                    "reward": td["next", "reward"].mean().item(),
                    "max step count": td["step_count"].max().item(),
                    **loss_dict,
                    **info_dict,
                    "batch": i,
                    "state distribution": wandb.Histogram(td["state"].argmax(dim=-1).cpu()),
                    "action distribution": wandb.Histogram(td["action"].argmax(dim=-1).cpu()),
                    "policy 'norm'": sum((p**2).sum() for p in agent.policy_module.parameters()),
                    "percentage_constraints_active": percentage_constraints_active,
                })
            if i % eval_every_n_batch == 0:
                with torch.no_grad(), set_exploration_type(InteractionType.DETERMINISTIC):
                    eval_data = env.rollout(100, agent.policy_module)
                eval_return = calc_return(eval_data["next", "reward"].flatten(), gamma)
                if log:
                    wandb.log({
                        "eval return": eval_return,
                        "batch": i
                    })
                eval_returns.append(eval_return)

            if progress_bar:
                pbar.update(td.numel())
    except Exception as e:
        print(f"Training interrupted due to an error: {e}")
        if progress_bar:
            pbar.close()
    return eval_returns
