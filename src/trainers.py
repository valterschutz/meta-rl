
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from tensordict.nn.probabilistic import InteractionType
from torch import multiprocessing, nn
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers import (ListStorage, PrioritizedSampler,
                                         ReplayBuffer)
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (Compose, DMControlEnv, DoubleToFloat,
                          ObservationNorm, ParallelEnv, StepCounter,
                          TransformedEnv, set_gym_backend)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.transforms import CatTensors
from torchrl.envs.utils import (ExplorationType, check_env_specs,
                                set_exploration_type)
from torchrl.modules import (OneHotCategorical, ProbabilisticActor, TanhNormal,
                             ValueOperator)
from torchrl.objectives import DiscreteSACLoss, SoftUpdate, ValueEstimators
from torchrl.objectives.value import GAE
from torchrl.record import VideoRecorder
from torchrl.record.loggers.csv import CSVLogger
from tqdm import tqdm

import wandb

sys.path.append(os.path.join(os.path.dirname(__file__), "./"))

from envs.toy_env import ToyEnv
from utils import calc_return
from agents.toy_agents import BaseAgent


def train_base_agent(device, total_frames, min_buffer_size, n_states, shortcut_steps, return_x, return_y, percentage_constraints_active, times_to_eval, log, progress_bar):
    env_max_steps = 5*n_states
    big_reward = 10.0
    gamma = 0.99
    batch_size = n_states*10
    sub_batch_size = 20
    num_epochs = 100
    lr = 5e-3
    target_eps = 0.99
    alpha = 0.7
    beta = 0.5
    max_grad_norm = 100.0

    n_batches = total_frames // batch_size
    eval_every_n_batch = n_batches // times_to_eval

    transforms = Compose(
        StepCounter(max_steps=env_max_steps),
    )
    x, y = ToyEnv.calculate_xy(n_states=n_states, shortcut_steps=shortcut_steps, return_x=return_x, return_y=return_y, big_reward=big_reward, gamma=gamma)
    env = ToyEnv(
        left_reward=x,
        right_reward=x,
        down_reward=y,
        up_reward=y,
        n_states=n_states,
        shortcut_steps=shortcut_steps,
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
                # Always use constrained return for evaluation
                eval_return = calc_return((eval_data["next", "normal_reward"]+eval_data["next","constraint_reward"]).flatten(), gamma)
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
