from torchrl.envs.utils import check_env_specs, step_mdp
from torchrl.modules.tensordict_module.actors import QValueActor
from torchrl.data import OneHot

from env import ToyEnv

from tensordict import TensorDict
import torch
from torch import nn

n_states = 7
n_actions = 4
env = ToyEnv(n_states)

td = env.reset()
print(td)

check_env_specs(env)

actor_net = nn.Linear(n_states, n_actions)
# qvalue_actor = QValueActor(module=actor_net, in_keys=["pos"], spec=OneHot(n_actions))
qvalue_actor = QValueActor(module=actor_net, in_keys=["pos"], action_space="one-hot")

n_batches = 5
# td = TensorDict({"observation": torch.randn(n_batches, n_states)}, [n_batches])
print(qvalue_actor(td))

# # Take 10 random actions and print out state sequence
# td = env.reset()
# for i in range(10):
#     td = env.rand_step(td)
#     # print(f"step {i}, state {td['next','state']}, reward {td['next','reward']}")
#     print(f"step {i}, td {td}")
#     td = step_mdp(td)
#     if td["done"]:
#         break
