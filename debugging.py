from base import get_base_from_config
from agents import MetaAgent
import json
import sys
from utils import DictWrapper
from env import MetaEnv

from torchrl.collectors import SyncDataCollector

with open(sys.argv[1], encoding="UTF-8") as f:
    config = json.load(f)

base_env, base_agent, base_collector = get_base_from_config(DictWrapper(config))
i = iter(base_collector)
td = next(i)
base_agent.process_batch(td)
td = next(i)
base_agent.process_batch(td)

# Try to replicate the error
base_agent.reset()
# collector.reset()
# i = iter(collector)
base_env.set_constraint_weight(0.5)
td = next(i)
base_agent.process_batch(td)

# Replicate error using meta env
meta_env = MetaEnv(
    base_env=base_env,
    base_agent=base_agent,
    base_collector=base_collector,
    device="cpu",
)

meta_td = meta_env.reset()
meta_td["action"] = 0
meta_td = meta_env.step(meta_td)

meta_td = meta_env.reset()
meta_td["action"] = 0.1
meta_td = meta_env.step(meta_td)

meta_td = meta_env.reset()
meta_td["action"] = 0.2
meta_td = meta_env.step(meta_td)

meta_agent = MetaAgent(
    state_spec=meta_env.state_spec,
    action_spec=meta_env.action_spec,
    num_optim_epochs=1,
    buffer_size=1,
    sub_batch_size=1,
    device="cpu",
    max_grad_norm=1,
    lr=1e-3,
    gamma=0.99,
    lmbda=0.96,
    clip_epsilon=0.2,
    use_entropy=True,
    hidden_units=8,
)

meta_collector = SyncDataCollector(
    meta_env,
    meta_agent.policy,
    frames_per_batch=1,
    total_frames=10,
    split_trajs=False,
    device="cpu",
)

print("here")
for meta_td in meta_collector:
    pass
