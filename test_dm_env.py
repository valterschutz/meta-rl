from torchrl.envs.libs.dm_control import DMControlEnv
import matplotlib.pyplot as plt

env = DMControlEnv("cartpole", "swingup_sparse")
td = env.reset()
print("result of reset: ", td)
env.close()

plt.imshow(td.get("pixels").numpy())
plt.show()
