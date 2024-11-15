from collections import defaultdict
from tqdm import tqdm
from datetime import datetime

from torchrl.envs.utils import check_env_specs, step_mdp
from torchrl.modules.tensordict_module.actors import QValueActor
from torchrl.modules import EGreedyModule
from torchrl.data import OneHot, ReplayBuffer
from torchrl.data.replay_buffers import LazyTensorStorage, SamplerWithoutReplacement
from torchrl.objectives import DQNLoss, SoftUpdate
from torchrl.collectors import SyncDataCollector
from torchrl.trainers import Trainer
from torchrl.record.loggers.csv import CSVLogger
from torchrl.record.loggers.wandb import WandbLogger


from tensordict import TensorDict
from tensordict.nn import TensorDictSequential
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import numpy as np

import wandb

from env import ToyEnv
from value_iteration import value_iteration


class OneHotLayer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        # Convert the integer state to one-hot encoded vector
        x_onehot = F.one_hot(x.to(torch.int64), num_classes=self.num_classes).float()
        return x_onehot


device = torch.device("cpu")
buffer_size = 1_000
total_frames = 1_000
frames_per_batch = total_frames
sub_batch_size = 1_000
num_optim_epochs = 1000
gamma = 1
n_actions = 4
max_grad_norm = 1
lr = 1e-1

logs = defaultdict(list)
pbar = tqdm(total=total_frames)

x = 1
y = 3
n_pos = 8
big_reward = 10
env = ToyEnv(x, y, n_pos, big_reward, random_start=True)

td = env.reset()

check_env_specs(env)

wandb.login()
wandb.init(
    project="toy_dqn",
    name=f"toy_dqn|{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    config={
        "buffer_size": buffer_size,
        "frames_per_batch": frames_per_batch,
        "sub_batch_size": sub_batch_size,
        "total_frames": total_frames,
        "num_optim_epochs": num_optim_epochs,
        "gamma": gamma,
        "n_actions": n_actions,
        "max_grad_norm": max_grad_norm,
        "lr": lr,
        "x": x,
        "y": y,
        "n_pos": n_pos,
        "big_reward": big_reward,
    },
)


actor_net = nn.Sequential(
    OneHotLayer(num_classes=n_pos), nn.Linear(n_pos, n_actions)
).to(device)

actor = QValueActor(module=actor_net, in_keys=["pos"], spec=env.action_spec)

exploration_module = EGreedyModule(
    spec=env.action_spec,
    eps_init=1.0,
    eps_end=1.0,
    annealing_num_steps=total_frames // 2,
)
actor_explore = TensorDictSequential(actor, exploration_module)

loss_module = DQNLoss(actor, action_space=env.action_spec)
loss_module.make_value_estimator(gamma=gamma)
target_updater = SoftUpdate(loss_module, eps=0.995)

optim = torch.optim.Adam(loss_module.parameters(), lr=lr)

replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(max_size=buffer_size),
    sampler=SamplerWithoutReplacement(),
)

collector = SyncDataCollector(
    env,
    actor_explore,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    split_trajs=False,
    device=device,
)

# add datetime to exp_name, without milliseconds
all_pos = torch.arange(n_pos)
td_all_pos = TensorDict(
    {"pos": all_pos, "params": env.params.expand(n_pos)},
    batch_size=[n_pos],
).to(device)

for i, td in enumerate(collector):
    replay_buffer.extend(td.reshape(-1).cpu())
    for _ in range(num_optim_epochs):
        subdata = replay_buffer.sample(sub_batch_size)
        loss_vals = loss_module(subdata.to(device))
        loss_value = loss_vals["loss"]  # TODO
        loss_value.backward()
        nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
        # save magnitude of gradients in logs
        logs["grad_norm"].append(
            sum(p.grad.data.norm(2).item() ** 2 for p in loss_module.parameters())
            ** 0.5
        )
        optim.step()
        optim.zero_grad()
        target_updater.step()
        wandb.log({"loss": loss_value.item(), "grad_norm": logs["grad_norm"][-1]})
    pbar.update(td.numel())
    exploration_module.step(td.numel())
    wandb.log(
        {
            "reward": td["next", "reward"].float().mean().item(),
            "epsilon": exploration_module.eps.item(),
        }
    )

# True Q-values are stored as "Q.pt"
true_q_values = value_iteration(x, y, n_pos, big_reward, gamma)[:-1]
# Visualize Q-values at the end of training
with torch.no_grad():
    thing = actor(td_all_pos["pos"])
    # print(f"{thing=}")
    q_values = thing[1].cpu().numpy()[:-1]
error_q_values = np.abs(true_q_values - q_values)

# Min and max values for the colorbar
min_val = min(q_values.min(), true_q_values.min(), error_q_values.min())
max_val = max(q_values.max(), true_q_values.max(), error_q_values.max())

fig, ax = plt.subplots()
cax = ax.matshow(q_values, cmap="inferno", vmin=min_val, vmax=max_val)
fig.colorbar(cax)
# colorbar limits
cax.set_clim(min_val, max_val)
fig.canvas.draw()
q_value_image = PILImage.frombytes(
    "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
)

fig, ax = plt.subplots()
cax = ax.matshow(true_q_values, cmap="inferno", vmin=min_val, vmax=max_val)
fig.colorbar(cax)
fig.canvas.draw()
true_q_value_image = PILImage.frombytes(
    "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
)

fig, ax = plt.subplots()
cax = ax.matshow(error_q_values, cmap="inferno", vmin=0, vmax=max_val)
fig.colorbar(cax)
fig.canvas.draw()
error_q_value_image = PILImage.frombytes(
    "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
)

wandb.log(
    {
        "Estimated Q-values": wandb.Image(q_value_image),
        "True Q-values": wandb.Image(true_q_value_image),
        "Error Q-values": wandb.Image(error_q_value_image),
    }
)


# plt.plot(logs["loss"])
# plt.title("Loss")
# plt.show()
# plt.plot(logs["grad_norm"])
# plt.title("Grad Norm")
# plt.show()
# plt.plot(logs["reward"])
# plt.title("Reward")
# plt.show()

# # Visualize Q-values
# pos = torch.arange(n_pos)
# td = TensorDict({"pos": pos}, batch_size=[n_pos])
# with torch.no_grad():
#     td = actor(td.to(device))
# q_values = td["action_value"]
# print(f"{q_values=}")
# plt.imshow(q_values.numpy(), cmap="hot", interpolation="nearest")
# # add colorbar
# plt.colorbar()
# plt.show()
