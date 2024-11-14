from collections import defaultdict
from tqdm import tqdm

from torchrl.envs.utils import check_env_specs, step_mdp
from torchrl.modules.tensordict_module.actors import QValueActor
from torchrl.modules import EGreedyModule
from torchrl.data import OneHot, ReplayBuffer
from torchrl.data.replay_buffers import LazyTensorStorage, SamplerWithoutReplacement
from torchrl.objectives import DQNLoss
from torchrl.collectors import SyncDataCollector

from env import ToyEnv

from tensordict import TensorDict
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class OneHotLayer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        # Convert the integer state to one-hot encoded vector
        return F.one_hot(x.to(torch.int64), num_classes=self.num_classes).float()


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
logs = defaultdict(list)
frames_per_batch = 100
sub_batch_size = 32
total_frames = 1000
num_epochs = 10
pbar = tqdm(total=total_frames)


n_states = 7
n_actions = 4
env = ToyEnv(n_states)

td = env.reset()

check_env_specs(env)


actor_net = nn.Sequential(
    OneHotLayer(num_classes=n_states), nn.Linear(n_states, n_actions)
).to(device)
# qvalue_actor = QValueActor(module=actor_net, in_keys=["pos"], spec=OneHot(n_actions))

actor = QValueActor(module=actor_net, in_keys=["pos"], action_space="categorical").to(
    device
)

actor_explore = nn.Sequential(
    actor,
    EGreedyModule(
        spec=env.action_spec,
        eps_init=1.0,
        eps_end=0.1,
        annealing_num_steps=total_frames,
    ),
)

# td = actor_explore(td)
# print(td)
# fail

loss_module = DQNLoss(actor, action_space="categorical")

td = actor_explore(td)
td = env.step(td)
loss = loss_module(td)
print(loss["loss"])
fail

optim = torch.optim.Adam(loss_module.parameters(), lr=1e-3)


replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(max_size=frames_per_batch),
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

for i, td in enumerate(collector):
    for _ in range(num_epochs):
        replay_buffer.extend(td.reshape(-1).cpu())
        for _ in range(frames_per_batch // sub_batch_size):
            subdata = replay_buffer.sample(sub_batch_size)
            loss_vals = loss_module(subdata.to(device))
            # print(loss_vals)
            loss_value = loss_vals["loss"]  # TODO
            loss_value.backward()
            optim.step()
            optim.zero_grad()
    pbar.update(td.numel())
    logs["reward"].append(td["next", "reward"].to(torch.float).mean().item())

plt.plot(logs["reward"])
plt.show()
