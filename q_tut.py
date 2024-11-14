import os
import uuid

import torch
from torch import nn
from torchrl.collectors import MultiaSyncDataCollector, SyncDataCollector
from torchrl.data import LazyMemmapStorage, MultiStep, TensorDictReplayBuffer
from torchrl.envs import (
    EnvCreator,
    ExplorationType,
    ParallelEnv,
    RewardScaling,
    StepCounter,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.transforms import (
    CatFrames,
    Compose,
    GrayScale,
    ObservationNorm,
    Resize,
    ToTensorImage,
    TransformedEnv,
)
from torchrl.modules import DuelingCnnDQNet, EGreedyModule, QValueActor

from torchrl.objectives import DQNLoss, SoftUpdate
from torchrl.record.loggers.csv import CSVLogger
from torchrl.trainers import (
    LogReward,
    Recorder,
    ReplayBufferTrainer,
    Trainer,
    UpdateWeights,
)


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def make_env(
    parallel=False,
    obs_norm_sd=None,
    num_workers=1,
):
    if obs_norm_sd is None:
        obs_norm_sd = {"standard_normal": True}
    if parallel:

        def maker():
            return GymEnv(
                "CartPole-v1",
                from_pixels=True,
                pixels_only=True,
                device=device,
            )

        base_env = ParallelEnv(
            num_workers,
            EnvCreator(maker),
            # Don't create a sub-process if we have only one worker
            serial_for_single=True,
            mp_start_method=mp_context,
        )
    else:
        base_env = GymEnv(
            "CartPole-v1",
            from_pixels=True,
            pixels_only=True,
            device=device,
        )

    env = TransformedEnv(
        base_env,
        Compose(
            StepCounter(),  # to count the steps of each trajectory
            ToTensorImage(),
            RewardScaling(loc=0.0, scale=0.1),
            GrayScale(),
            Resize(64, 64),
            CatFrames(4, in_keys=["pixels"], dim=-3),
            ObservationNorm(in_keys=["pixels"], **obs_norm_sd),
        ),
    )
    return env


def get_norm_stats():
    test_env = make_env()
    test_env.transform[-1].init_stats(
        num_iter=1000, cat_dim=0, reduce_dim=[-1, -2, -4], keep_dims=(-1, -2)
    )
    obs_norm_sd = test_env.transform[-1].state_dict()
    # let's check that normalizing constants have a size of ``[C, 1, 1]`` where
    # ``C=4`` (because of :class:`~torchrl.envs.CatFrames`).
    print("state dict of the observation norm:", obs_norm_sd)
    test_env.close()
    del test_env
    return obs_norm_sd


def make_model(dummy_env):
    cnn_kwargs = {
        "num_cells": [32, 64, 64],
        "kernel_sizes": [6, 4, 3],
        "strides": [2, 2, 1],
        "activation_class": nn.ELU,
        # This can be used to reduce the size of the last layer of the CNN
        # "squeeze_output": True,
        # "aggregator_class": nn.AdaptiveAvgPool2d,
        # "aggregator_kwargs": {"output_size": (1, 1)},
    }
    mlp_kwargs = {
        "depth": 2,
        "num_cells": [
            64,
            64,
        ],
        "activation_class": nn.ELU,
    }
    net = DuelingCnnDQNet(
        dummy_env.action_spec.shape[-1], 1, cnn_kwargs, mlp_kwargs
    ).to(device)
    net.value[-1].bias.data.fill_(init_bias)

    actor = QValueActor(net, in_keys=["pixels"], spec=dummy_env.action_spec).to(device)
    # init actor: because the model is composed of lazy conv/linear layers,
    # we must pass a fake batch of data through it to instantiate them.
    tensordict = dummy_env.fake_tensordict()
    actor(tensordict)

    # we join our actor with an EGreedyModule for data collection
    exploration_module = EGreedyModule(
        spec=dummy_env.action_spec,
        annealing_num_steps=total_frames,
        eps_init=eps_greedy_val,
        eps_end=eps_greedy_val_env,
    )
    actor_explore = TensorDictSequential(actor, exploration_module)

    return actor, actor_explore


def get_replay_buffer(buffer_size, n_optim, batch_size):
    replay_buffer = TensorDictReplayBuffer(
        batch_size=batch_size,
        storage=LazyMemmapStorage(buffer_size),
        prefetch=n_optim,
    )
    return replay_buffer


def get_collector(
    stats,
    num_collectors,
    actor_explore,
    frames_per_batch,
    total_frames,
    device,
):
    # We can't use nested child processes with mp_start_method="fork"
    if is_fork:
        cls = SyncDataCollector
        env_arg = make_env(parallel=True, obs_norm_sd=stats, num_workers=num_workers)
    else:
        cls = MultiaSyncDataCollector
        env_arg = [
            make_env(parallel=True, obs_norm_sd=stats, num_workers=num_workers)
        ] * num_collectors
    data_collector = cls(
        env_arg,
        policy=actor_explore,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        # this is the default behavior: the collector runs in ``"random"`` (or explorative) mode
        exploration_type=ExplorationType.RANDOM,
        # We set the all the devices to be identical. Below is an example of
        # heterogeneous devices
        device=device,
        storing_device=device,
        split_trajs=False,
        postproc=MultiStep(gamma=gamma, n_steps=5),
    )
    return data_collector


def get_loss_module(actor, gamma):
    loss_module = DQNLoss(actor, delay_value=True)
    loss_module.make_value_estimator(gamma=gamma)
    target_updater = SoftUpdate(loss_module, eps=0.995)
    return loss_module, target_updater


is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)

# the learning rate of the optimizer
lr = 2e-3
# weight decay
wd = 1e-5
# the beta parameters of Adam
betas = (0.9, 0.999)
# Optimization steps per batch collected (aka UPD or updates per data)
n_optim = 8

gamma = 0.99
tau = 0.02
