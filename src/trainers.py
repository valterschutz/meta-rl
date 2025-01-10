
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict import TensorDict, TensorDictBase
from tensordict.nn.distributions import NormalParamExtractor
from tensordict.nn.probabilistic import InteractionType
from torch import multiprocessing, nn
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer, BoundedTensorSpec
from torchrl.data.replay_buffers import (ListStorage, PrioritizedSampler,
                                         ReplayBuffer)
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (Compose, DMControlEnv, DoubleToFloat,
                          ObservationNorm, ParallelEnv, StepCounter,
                          TransformedEnv, set_gym_backend)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.transforms import CatTensors, Transform, RenameTransform, DoubleToFloat, DTypeCastTransform
from torchrl.envs.transforms.transforms import _apply_to_composite
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
from agents.base_agents import ToySACAgent, PointBaseAgent, ReacherBaseAgent


def train_toy_base_agent(device, total_frames, min_buffer_size, n_states, shortcut_steps, return_x, return_y, when_constraints_active, times_to_eval, log, progress_bar, batch_size, sub_batch_size, num_epochs):
    """
    Train a base agent in the toy environment.
    """
    env_max_steps = 5*n_states
    # env_max_steps = total_frames
    big_reward = 10.0
    gamma = 0.99
    actor_lr = 3e-4
    critic_lr = 3e-4
    alpha_lr = 3e-4
    target_eps = 0.995
    alpha = 0.7
    beta = 0.5
    max_grad_norm = 100.0

    n_batches = total_frames // batch_size
    eval_every_n_batch = n_batches // times_to_eval

    transforms = Compose(
        StepCounter(max_steps=env_max_steps),
        DTypeCastTransform(dtype_in=torch.long, dtype_out=torch.float32, in_keys=["observation"])
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

    agent = ToySACAgent(
        state_spec=env.state_spec,
        action_spec=env.action_spec,
        device=device,
        buffer_size = total_frames,
        min_buffer_size = min_buffer_size,
        batch_size = batch_size,
        sub_batch_size = sub_batch_size,
        num_epochs = num_epochs,
        actor_lr = actor_lr,
        critic_lr = critic_lr,
        alpha_lr = alpha_lr,
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
    if isinstance(when_constraints_active, float):
        batch_to_activate_constraints = int(n_batches * when_constraints_active)
    for td in rand_collector:
        agent.process_batch(td, constraints_active=False)
        if progress_bar:
            pbar.update(td.numel())

    eval_returns = []
    try:
        for i, td in enumerate(collector):
            td["action"] = td["action"].to(torch.float32) # Due to bug in torchrl, need to manually cast to float
            collector.update_policy_weights_() # Check if this is necessary

            # Constraints are either deterministically set at some batch or decided by a callback function
            if isinstance(when_constraints_active, float):
                constraints_active: bool = i >= batch_to_activate_constraints
            elif callable(when_constraints_active):
                constraints_active: bool = when_constraints_active(td)

            loss_dict, info_dict = agent.process_batch(td, constraints_active=constraints_active)

            if log:
                wandb.log({
                    "normal_reward": td["next", "normal_reward"].mean().item(),
                    "constraint_reward": td["next", "constraint_reward"].mean().item(),
                    "reward": (td["next", "normal_reward"] + td["next", "constraint_reward"]).mean().item(),
                    "max step count": td["step_count"].max().item(),
                    **loss_dict,
                    **info_dict,
                    "batch": i,
                    "next state distribution": wandb.Histogram(td["next","state"].argmax(dim=-1).cpu()+1),
                    "action distribution": wandb.Histogram(td["action"].argmax(dim=-1).cpu()+1),
                    "policy 'norm'": sum((p**2).sum() for p in agent.policy_module.parameters()),
                    "constraints_active": float(constraints_active)
                })
            if i % eval_every_n_batch == 0:
                with torch.no_grad(), set_exploration_type(InteractionType.DETERMINISTIC):
                    eval_data = env.rollout(1000, agent.policy_module)
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
    except KeyboardInterrupt as e:
        print(f"Training interrupted due to an error: {e}")
        if progress_bar:
            pbar.close()
    return eval_returns

def train_point_base_agent(device, total_frames, min_buffer_size, when_constraints_active, times_to_eval, log, progress_bar, batch_size, sub_batch_size, num_epochs):
    """
    Train a base agent in the Point environment.
    """
    env_max_steps = total_frames
    gamma = 0.99
    lr = 5e-3
    target_eps = 0.99
    alpha = 0.7
    beta = 0.5
    max_grad_norm = 100.0

    n_batches = total_frames // batch_size
    eval_every_n_batch = n_batches // times_to_eval

    # Custom transform for adding constraints
    class NegativeNormTransform(Transform):
        def _apply_transform(self, t: torch.Tensor) -> None:
            return -t.norm().unsqueeze(-1)

        # The transform must also modify the data at reset time
        def _reset(
            self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
        ) -> TensorDictBase:
            return self._call(tensordict_reset)

        @_apply_to_composite
        def transform_observation_spec(self, observation_spec):
            return BoundedTensorSpec(
                low=-1,
                high=1,
                shape=observation_spec.shape,
                dtype=observation_spec.dtype,
                device=observation_spec.device,
            )

    transforms = Compose(
        DoubleToFloat(),
        StepCounter(max_steps=env_max_steps),
        CatTensors(in_keys=["position", "velocity"], out_key="state", del_keys=False),
        RenameTransform(in_keys=["reward"], out_keys=["normal_reward"], create_copy=True),
        NegativeNormTransform(in_keys=["velocity"], out_keys=["constraint_reward"]),
    )
    env = DMControlEnv("point_mass", "easy")
    env = TransformedEnv(
        env,
        transforms
    )

    agent = PointBaseAgent(
        env.action_spec,
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
    if isinstance(when_constraints_active, float):
        batch_to_activate_constraints = int(n_batches * when_constraints_active)
    for td in rand_collector:
        agent.process_batch(td, constraints_active=False)
        if progress_bar:
            pbar.update(td.numel())

    eval_returns = []
    try:
        for i, td in enumerate(collector):
            td["action"] = td["action"].to(torch.float32) # Due to bug in torchrl, need to manually cast to float
            collector.update_policy_weights_()

            # Constraints are either deterministically set at some batch or decided by a callback function
            if isinstance(when_constraints_active, float):
                constraints_active: bool = i >= batch_to_activate_constraints
            elif callable(when_constraints_active):
                constraints_active: bool = when_constraints_active(td)

            loss_dict, info_dict = agent.process_batch(td, constraints_active=constraints_active)

            if log:
                wandb.log({
                    "normal_reward": td["next", "normal_reward"].mean().item(),
                    "constraint_reward": td["next", "constraint_reward"].mean().item(),
                    "reward": (td["next", "normal_reward"] + td["next", "constraint_reward"]).mean().item(),
                    "max step count": td["step_count"].max().item(),
                    **loss_dict,
                    **info_dict,
                    "batch": i,
                    "pos x distribution": wandb.Histogram(td["position"][0].cpu()),
                    "pos y distribution": wandb.Histogram(td["position"][1].cpu()),
                    "vel x distribution": wandb.Histogram(td["velocity"][0].cpu()),
                    "vel y distribution": wandb.Histogram(td["velocity"][1].cpu()),
                    "x action": wandb.Histogram(td["action"][0].cpu()),
                    "y action": wandb.Histogram(td["action"][1].cpu()),
                    "policy 'norm'": sum((p**2).sum() for p in agent.policy_module.parameters()),
                    "constraints_active": float(constraints_active)
                })
            if i % eval_every_n_batch == 0:
                with torch.no_grad(), set_exploration_type(InteractionType.DETERMINISTIC):
                    eval_data = env.rollout(10_000, agent.policy_module)
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
    except KeyboardInterrupt as e:
        print(f"Training interrupted due to an error: {e}")
        # print(f"{td["next", "reward"].shape=} {td["done"].shape=}, {}")
        if progress_bar:
            pbar.close()
    return eval_returns

def train_reacher_base_agent(device, total_frames, min_buffer_size, when_constraints_active, times_to_eval, log, progress_bar, batch_size, sub_batch_size, num_epochs):
    """
    Train a base agent in the Point environment.
    """
    env_max_steps = total_frames
    gamma = 0.99
    lr = 5e-3
    target_eps = 0.99
    alpha = 0.7
    beta = 0.5
    max_grad_norm = 100.0

    n_batches = total_frames // batch_size
    eval_every_n_batch = n_batches // times_to_eval

    # Custom transform for adding constraints
    class NegativeNormTransform(Transform):
        def _apply_transform(self, t: torch.Tensor) -> None:
            return -t.norm().unsqueeze(-1)

        # The transform must also modify the data at reset time
        def _reset(
            self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
        ) -> TensorDictBase:
            return self._call(tensordict_reset)

        @_apply_to_composite
        def transform_observation_spec(self, observation_spec):
            return BoundedTensorSpec(
                low=-1,
                high=1,
                shape=observation_spec.shape,
                dtype=observation_spec.dtype,
                device=observation_spec.device,
            )

    transforms = Compose(
        DoubleToFloat(),
        StepCounter(max_steps=env_max_steps),
        CatTensors(in_keys=["position", "velocity"], out_key="state", del_keys=False),
        RenameTransform(in_keys=["reward"], out_keys=["normal_reward"], create_copy=True),
        NegativeNormTransform(in_keys=["velocity"], out_keys=["constraint_reward"]),
    )
    env = DMControlEnv("reacher", "easy")
    env = TransformedEnv(
        env,
        transforms
    )

    agent = ReacherBaseAgent(
        env.action_spec,
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
    if isinstance(when_constraints_active, float):
        batch_to_activate_constraints = int(n_batches * when_constraints_active)
    for td in rand_collector:
        agent.process_batch(td, constraints_active=False)
        if progress_bar:
            pbar.update(td.numel())

    eval_returns = []
    try:
        for i, td in enumerate(collector):
            td["action"] = td["action"].to(torch.float32) # Due to bug in torchrl, need to manually cast to float
            collector.update_policy_weights_()

            # Constraints are either deterministically set at some batch or decided by a callback function
            if isinstance(when_constraints_active, float):
                constraints_active: bool = i >= batch_to_activate_constraints
            elif callable(when_constraints_active):
                constraints_active: bool = when_constraints_active(td)

            loss_dict, info_dict = agent.process_batch(td, constraints_active=constraints_active)

            if log:
                wandb.log({
                    "normal_reward": td["next", "normal_reward"].mean().item(),
                    "constraint_reward": td["next", "constraint_reward"].mean().item(),
                    "reward": (td["next", "normal_reward"] + td["next", "constraint_reward"]).mean().item(),
                    "max step count": td["step_count"].max().item(),
                    **loss_dict,
                    **info_dict,
                    "batch": i,
                    "pos x distribution": wandb.Histogram(td["position"][0].cpu()),
                    "pos y distribution": wandb.Histogram(td["position"][1].cpu()),
                    "vel x distribution": wandb.Histogram(td["velocity"][0].cpu()),
                    "vel y distribution": wandb.Histogram(td["velocity"][1].cpu()),
                    "x action": wandb.Histogram(td["action"][0].cpu()),
                    "y action": wandb.Histogram(td["action"][1].cpu()),
                    "policy 'norm'": sum((p**2).sum() for p in agent.policy_module.parameters()),
                    "constraints_active": float(constraints_active)
                })
            if i % eval_every_n_batch == 0:
                with torch.no_grad(), set_exploration_type(InteractionType.DETERMINISTIC):
                    eval_data = env.rollout(10_000, agent.policy_module)
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
    except KeyboardInterrupt as e:
        print(f"Training interrupted due to an error: {e}")
        # print(f"{td["next", "reward"].shape=} {td["done"].shape=}, {}")
        if progress_bar:
            pbar.close()
    return eval_returns

def train_meta_agent(
    device,
    n_base_episodes,
    log,
    progress_bar,
    batch_size,
    sub_batch_size,
    num_epochs
):
    meta_agent = MetaAgent(state_keys, device, buffer_size, min_buffer_size, sub_batch_size, num_epochs, lr, gamma, target_eps, alpha, beta, max_grad_norm)
    # Define the meta callback, a function that takes the base agents tensordict and returns a boolean that indicates if
    # the constraints should be activated
    def meta_callback(base_td):
        return True
    eval_returns = []
    try:
        for i in range(n_base_episodes):
            base_returns = train_base_agent(
                device=torch.device("cpu"),
                total_frames=50_000,
                min_buffer_size=0,
                n_states=n_states,
                shortcut_steps=5,
                return_x=return_x,
                return_y=-100,
                when_constraints_active=meta_callback,
                times_to_eval=20,
                log=True,
                progress_bar=True,
                batch_size = 200,
                sub_batch_size = 20,
                num_epochs = 100
            )

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
                    "when_constraints_active": when_constraints_active if isinstance(when_constraints_active, float) else 0.0,
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
