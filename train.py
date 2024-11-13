from torchrl.envs.utils import check_env_specs, step_mdp

from env import ToyEnv

env = ToyEnv()
check_env_specs(env)

# Take 10 random actions and print out state sequence
td = env.reset()
for i in range(10):
    td = env.rand_step(td)
    print(f"step {i}, state {td['next','state']}, reward {td['next','reward']}")
    td = step_mdp(td)
    if td["done"]:
        break
