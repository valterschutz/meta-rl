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
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.record import VideoRecorder
from torchrl.record.loggers.csv import CSVLogger
from tqdm import tqdm

import wandb


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


device = (
    torch.device(0)
    if torch.cuda.is_available()
    else torch.device("cpu")
)
num_cells = 256
lr = 3e-4
# lr = 3e-6
# max_grad_norm = 1.0
max_grad_norm = 100.0

frames_per_batch = 1000
total_frames = 1_000_000
eval_every_n_batch = 10

sub_batch_size = 64
num_epochs = 10
clip_epsilon = 0.2
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4

transforms = Compose(
    CatTensors(in_keys=["position", "velocity"], out_key="state", del_keys=False),
    DoubleToFloat(),
    StepCounter(),
)
env = DMControlEnv("point_mass", "easy", device=device)
env = TransformedEnv(
    env,
    transforms
)
check_env_specs(env)

pixel_env = DMControlEnv("point_mass", "easy", device=device, from_pixels=True, pixels_only=False)
pixel_env = TransformedEnv(
    pixel_env,
    transforms,
)
check_env_specs(pixel_env)



rollout = env.rollout(3)

actor_net = nn.Sequential(
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(2 * env.action_spec.shape[-1], device=device),
    NormalParamExtractor(),
)

policy_module = TensorDictModule(
    actor_net, in_keys=["state"], out_keys=["loc", "scale"]
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
)
policy_module(env.reset())

value_net = nn.Sequential(
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(1, device=device),
)

value_module = ValueOperator(
    module=value_net,
    in_keys=["state"],
)
value_module(env.reset())


collector = SyncDataCollector(
    env,
    policy_module,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    split_trajs=False,
    device=device,
)

replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(max_size=frames_per_batch),
    sampler=SamplerWithoutReplacement(),
)

advantage_module = GAE(
    gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True
)

loss_module = ClipPPOLoss(
    actor_network=policy_module,
    critic_network=value_module,
    # clip_epsilon=clip_epsilon,
    # entropy_bonus=bool(entropy_eps),
    # entropy_coef=entropy_eps,
    # critic_coef=1.0,
    # loss_critic_type="smooth_l1",
)
loss_keys = ["loss_objective", "loss_critic", "loss_entropy"]

optim = torch.optim.Adam(loss_module.parameters(), lr)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     optim, total_frames // frames_per_batch, 0.0
# )

logs = defaultdict(list)
pbar = tqdm(total=total_frames)
eval_str = ""

wandb.init(project="temp")

try:
    for i, tensordict_data in enumerate(collector):
        # Artificially add reward to the data, proportional to the distance from the origin
        # tensordict_data["next", "reward"] -= 0.1 * tensordict_data["position"].norm()
        for _ in range(num_epochs):
            advantage_module(tensordict_data)
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view.cpu())
            losses = {loss_key: [] for loss_key in loss_keys}
            grads = []
            for _ in range(frames_per_batch // sub_batch_size):
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

        losses = {loss_key: sum(losses[loss_key]) / len(losses[loss_key]) for loss_key in loss_keys}

        wandb.log({
            "reward": tensordict_data["next", "reward"].mean().item(),
            "max step count": tensordict_data["step_count"].max().item(),
            **losses,
            "mean gradient norm": sum(grads) / len(grads),
            "batch": i,
        })

        if i % eval_every_n_batch == 0:
            save_video(pixel_env, policy_module, {
                "batch": i
            })

        pbar.update(tensordict_data.numel())
        # scheduler.step()
except Exception as e:
    print(f"Training interrupted due to an error: {e}")
    pbar.close()

# Remove old environment and value module
del env
del value_module
del collector
del replay_buffer
