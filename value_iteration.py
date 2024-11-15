from env import ToyEnv
import torch
from tensordict import TensorDict


def value_iteration(x, y, n_pos, big_reward, gamma, tol=1e-6):
    n_actions = 4
    batch_size = n_pos * n_actions
    pos = (
        torch.arange(n_pos)[:, None].repeat(1, n_actions).reshape(-1)
    )  # [0, 0, ..., 0, 1, 1, ..., 1, ...]
    action = torch.arange(n_actions).repeat(
        n_pos
    )  # [0, 1, 2, 3, 0, 1, 2, 3, ...]  # [0, 0, ..., 0, 1, 1, ..., 1, ...]
    x = torch.tensor([1]).repeat(batch_size)
    y = torch.tensor([3]).repeat(batch_size)

    td = TensorDict(
        {
            "pos": pos,
            "action": action,
            "params": {
                "x": x * torch.ones_like(pos),
                "y": y * torch.ones_like(pos),
                "n_pos": n_pos * torch.ones_like(pos),
                "big_reward": big_reward * torch.ones_like(pos),
            },
        },
        batch_size=[batch_size],
    )

    env = ToyEnv(x, y, n_pos, big_reward)

    next_td = env.step(td)

    # Perform value iteration
    Q = torch.zeros((n_pos, n_actions))
    delta = 1
    while delta > tol:
        delta = 0
        for s in range(n_pos - 1):  # Final state has value 0 since it's terminal
            for a in range(n_actions):
                idx = s * n_actions + a
                next_pos = next_td["next", "pos"][idx].item()
                reward = next_td["next", "reward"][idx].item()
                old_Q = Q[s, a].item()
                Q[s, a] = reward + gamma * Q[next_pos, :].max().item()
                delta = max(delta, abs(old_Q - Q[s, a]))
    return Q
