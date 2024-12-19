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

import wandb

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.envs.toy_env import ToyEnv
from src.utils import calc_return


def save_video(pixel_env, policy_module, log_dict):
    exp_name = "temp_exp"
    logger = CSVLogger(exp_name)
    recorder = VideoRecorder(logger, tag="my_video")
    record_env = TransformedEnv(pixel_env, recorder)
    with torch.no_grad(), set_exploration_type(InteractionType.DETERMINISTIC):
        rollout = record_env.rollout(max_steps=1000, policy=policy_module)
    recorder.dump() # Saves video as a .pt file at `csv`
    # Now load the file as a tensor
    video = torch.load(Path(logger.log_dir) / exp_name / "videos" / "my_video_0.pt").numpy()
    wandb.log({"video": wandb.Video(video), **log_dict})


device = torch.device("cpu")
num_cells = 20
lr = 1e-3
max_grad_norm = 100.0

frames_per_batch = 1000
total_frames = 1_000_000
eval_every_n_batch = 100

buffer_size = total_frames
min_buffer_size = 10_000
sub_batch_size = 100
num_epochs = 10
gamma = 0.99
target_eps = 0.99
lmbda=0.9

n_states = 20

transforms = Compose(
    StepCounter(max_steps=100),
)
x, y = ToyEnv.calculate_xy(n_states=n_states, return_x=5, return_y=1, big_reward=10, gamma=0.99)
env = ToyEnv(
    left_reward=x,
    right_reward=x,
    down_reward=y,
    up_reward=y,
    n_states=n_states,
    big_reward=10.0,
    constraints_active=False,
    random_start=False,
    seed=None,
    device=device
)
env = TransformedEnv(
    env,
    transforms
)
check_env_specs(env)

pixel_env = None



rollout = env.rollout(3)

n_actions = env.action_spec.shape[-1]

actor_net = nn.Sequential(
    nn.Linear(n_states, num_cells, device=device),
    nn.ReLU(),
    nn.Linear(num_cells, n_actions, device=device),
    nn.Sigmoid()
)

policy_module = TensorDictModule(
    actor_net, in_keys=["state"], out_keys=["probs"]
)

policy_module = ProbabilisticActor(
    module=policy_module,
    spec=env.action_spec,
    in_keys=["probs"],
    distribution_class=OneHotCategorical,
    return_log_prob=True,
)

qvalue_net = nn.Sequential(
    nn.Linear(n_states, num_cells, device=device),
    nn.ReLU(),
    nn.Linear(num_cells, n_actions, device=device),
)
qvalue_module = ValueOperator(
    module=qvalue_net,
    in_keys=["state"],
    out_keys=["action_value"],
)


collector = SyncDataCollector(
    env,
    policy_module,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    split_trajs=False,
    device=device,
)

replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(max_size=buffer_size),
    sampler=SamplerWithoutReplacement(),
)

# advantage_module = GAE(
#     gamma=gamma, lmbda=lmbda, value_network=qvalue_module, average_gae=True
# )

loss_module = DiscreteSACLoss(
    actor_network=policy_module,
    qvalue_network=qvalue_module,
    num_actions=n_actions,
    target_entropy=0.0, # TODO: does this work?
)

# loss_module.make_value_estimator(ValueEstimators.TDLambda, gamma=gamma, lmbda=lmbda)
loss_module.make_value_estimator(ValueEstimators.TD0, gamma=gamma)
loss_keys = ["loss_actor", "loss_qvalue", "loss_alpha"]

target_net = SoftUpdate(loss_module, eps=target_eps)

optim = torch.optim.Adam(loss_module.parameters(), lr)

logs = defaultdict(list)
pbar = tqdm(total=total_frames)
eval_str = ""

wandb.init(project="clean-base-sac")

try:
    for i, tensordict_data in enumerate(collector):
        collector.update_policy_weights_() # Check if this is necessary
        # Artificially add reward to the data, proportional to the distance from the origin
        # tensordict_data["next", "reward"] -= 0.1 * tensordict_data["position"].norm()

        # Add data to the replay buffer
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view.cpu())
        if len(replay_buffer) < min_buffer_size:
            continue
        for _ in range(num_epochs):
            losses = {loss_key: [] for loss_key in loss_keys}
            grads = []

            subdata = replay_buffer.sample(sub_batch_size)
            loss_vals = loss_module(subdata.to(device))
            loss = 0
            for loss_key in loss_keys:
                loss += loss_vals[loss_key]
                losses[loss_key].append(loss_vals[loss_key].item())

            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
            grads.append(grad)
            optim.step()
            optim.zero_grad()

            # Update target network
            target_net.step()

        losses = {loss_key: sum(losses[loss_key]) / len(losses[loss_key]) for loss_key in loss_keys}

        wandb.log({
            "reward": tensordict_data["next", "reward"].mean().item(),
            "max step count": tensordict_data["step_count"].max().item(),
            **losses,
            "mean gradient norm": sum(grads) / len(grads),
            "batch": i,
            "state distribution": wandb.Histogram(tensordict_data["state"].argmax(dim=-1)),
            "action distribution": wandb.Histogram(tensordict_data["action"].argmax(dim=-1)),
            "policy 'norm'": sum((p**2).sum() for p in policy_module.parameters())
        })

        if i % eval_every_n_batch == 0:
            with torch.no_grad(), set_exploration_type(InteractionType.DETERMINISTIC):
                eval_data = env.rollout(100, policy_module)
            eval_return = calc_return(eval_data["next", "reward"].flatten(), gamma)
            wandb.log({
                "eval return": eval_return,
                "batch": i
            })

        pbar.update(tensordict_data.numel())
        # scheduler.step()
except Exception as e:
    print(f"Training interrupted due to an error: {e}")
    pbar.close()

# Remove old environment and value module
del env
del qvalue_module
del collector
del replay_buffer
