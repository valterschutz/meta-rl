import matplotlib.pyplot as plt
import multiprocessing
from collections import defaultdict
from datetime import datetime

import torch
import wandb
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    ParallelEnv,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.libs.dm_control import DMControlEnv
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.envs.transforms import CatTensors
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.record import CSVLogger, VideoRecorder
from tqdm import tqdm

# torch.set_default_dtype(torch.double)

device = torch.device(1)


env = GymEnv("InvertedDoublePendulum-v4", device=device)
env = TransformedEnv(
    env,
    Compose(
        ObservationNorm(in_keys=["observation"]),
        DoubleToFloat(),
        StepCounter(),
    ),
)
print(f"Normalizing observation space... ", end="")
env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
print(f"Done.")

logger = CSVLogger("my_exp", video_format="mp4", video_fps=60)
record_env = GymEnv(
    "InvertedDoublePendulum-v4", from_pixels=True, pixels_only=False, device=device
)
recorder = VideoRecorder(logger, tag="my_video")
record_env = TransformedEnv(
    record_env,
    Compose(
        ObservationNorm(in_keys=["observation"]),
        DoubleToFloat(),
        recorder,
    ),
)
print(f"Normalizing observation space for record env... ", end="")
record_env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
print(f"Done.")

# is_fork = multiprocessing.get_start_method() == "fork"
num_cells = 256  # number of cells in each layer i.e. output dim.
lr = 3e-4
max_grad_norm = 1.0

batch_size = 1000
buffer_size = batch_size  # due to on-policy
# For a complete training, bring the number of frames up to 1M
total_frames = 100_000

sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
num_optim_epochs = 10  # optimization steps per batch of data collected
clip_epsilon = (
    0.2  # clip value for PPO loss: see the equation in the intro for more context.
)
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4
eval_every_n_batches = 5


# print("normalization constant shape:", env.transform[0].loc.shape)

# print("observation_spec:", env.observation_spec)
# print("reward_spec:", env.reward_spec)
# print("input_spec:", env.input_spec)
# print("action_spec (as defined by input_spec):", env.action_spec)

check_env_specs(env)

# rollout = env.rollout(3)
# print("rollout of three steps:", rollout)
# print("Shape of the rollout TensorDict:", rollout.batch_size)

actor_net = nn.Sequential(
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(2 * env.action_spec.shape[-1], device=device),
    NormalParamExtractor(),
).to(device)

policy_module = TensorDictModule(
    actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
)

policy_module = ProbabilisticActor(
    module=policy_module,
    spec=env.action_spec,
    in_keys=["loc", "scale"],
    distribution_class=TanhNormal,
    distribution_kwargs={
        "low": env.action_spec.space.low,
        "high": env.action_spec.space.high,
    },
    return_log_prob=True,
    # we'll need the log-prob for the numerator of the importance weights
)

value_net = nn.Sequential(
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(1, device=device),
).to(device)

value_module = ValueOperator(
    module=value_net,
    in_keys=["observation"],
)

# print("Running policy:", policy_module(env.reset()))
# print("Running value:", value_module(env.reset()))
policy_module(env.reset())
value_module(env.reset())

collector = SyncDataCollector(
    env,
    policy_module,
    frames_per_batch=batch_size,
    total_frames=total_frames,
    split_trajs=False,
    device=device,
)

replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(max_size=buffer_size),
    sampler=SamplerWithoutReplacement(),
)

advantage_module = GAE(
    gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True
)

loss_module = ClipPPOLoss(
    actor_network=policy_module,
    critic_network=value_module,
    clip_epsilon=clip_epsilon,
    entropy_bonus=bool(entropy_eps),
    entropy_coef=entropy_eps,
    # these keys match by default but we set this for completeness
    critic_coef=1.0,
    loss_critic_type="smooth_l1",
)

optim = torch.optim.Adam(loss_module.parameters(), lr)

# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     optim, total_frames // batch_size, 0.0
# )

logs = defaultdict(list)
pbar = tqdm(total=total_frames)

wandb.login()
wandb.init(
    project=f"meta_gym",
    name=f"meta_gym|{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    config={
        "buffer_size": buffer_size,
        "batch_size": batch_size,
        "sub_batch_size": sub_batch_size,
        "total_frames": total_frames,
        "num_optim_epochs": num_optim_epochs,
        "gamma": gamma,
        "max_grad_norm": max_grad_norm,
        "lr": lr,
    },
)

# We iterate over the collector until it reaches the total number of frames it was
# designed to collect:
for i, td in enumerate(collector):
    # we now have a batch of data to work with. Let's learn something from it.
    for _ in range(num_optim_epochs):
        replay_buffer.extend(td)  # reset replay buffer
        # We'll need an "advantage" signal to make PPO work.
        # We re-compute it at each epoch as its value depends on the value
        # network which is updated in the inner loop.
        advantage_module(td)
        # data_view = tensordict_data.reshape(-1)
        # replay_buffer.extend(data_view.cpu())
        epoch_losses = torch.zeros(batch_size // sub_batch_size, device=device)
        epoch_gradient_magnitudes = torch.zeros(
            batch_size // sub_batch_size, device=device
        )
        for j in range(batch_size // sub_batch_size):
            subdata = replay_buffer.sample(sub_batch_size)
            loss_vals = loss_module(subdata.to(device))
            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )
            epoch_losses[j] = loss_value
            # Optimization: backward, grad clipping and optimization step
            loss_value.backward()
            # this is not strictly mandatory but it's good practice to keep
            # your gradient norm bounded
            # torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
            epoch_gradient_magnitudes[j] = torch.sqrt(
                sum(
                    p.grad.norm() ** 2
                    for p in loss_module.parameters()
                    if p.grad is not None
                )
            )
            optim.step()
            optim.zero_grad()

    wandb.log(
        {
            "step_count": td["step_count"].max().item(),
            "lr": optim.param_groups[0]["lr"],
            "loss": epoch_losses.mean(),
            "gradient magnitude": epoch_gradient_magnitudes.mean(),
        }
    )
    pbar.update(td.numel())

    if i % eval_every_n_batches == 0:
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            eval_rollout = env.rollout(1000, policy_module)
            wandb.log(
                {
                    "eval reward (sum)": eval_rollout["next", "reward"].sum().item(),
                    "eval step_count": eval_rollout["step_count"].max().item(),
                }
            )
            del eval_rollout

    # We're also using a learning rate scheduler. Like the gradient clipping,
    # this is a nice-to-have but nothing necessary for PPO to work.
    # scheduler.step()

# Save a video of the trained policy
with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
    record_env.rollout(1000, policy_module)
record_env.transform[-1].dump()

env.close()
record_env.close()
