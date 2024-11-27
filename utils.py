import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb


class OneHotLayer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        # Convert the integer state to one-hot encoded vector
        x_onehot = F.one_hot(x.to(torch.int64), num_classes=self.num_classes).float()
        return x_onehot


def log(pbar, meta_td, base_eval_td, episode, step):
    pbar.update(meta_td.numel())
    wandb.log(
        {
            f"episode-{episode}/step": step,
            # "base reward": base_td["next", "reward"].float().mean().item(),
            # "base loss_objective": base_loss_vals["loss_objective"].item(),
            # "base loss_critic": base_loss_vals["loss_critic"].item(),
            # "base loss": base_loss.item(),
            # "base state distribution": wandb.Histogram(
            # base_td["pos"].cpu().numpy()
            # ),
            # "base reward distribution": wandb.Histogram(
            #     base_td["next", "reward"].cpu().numpy()
            # ),
            # "state distribution": wandb.Histogram(np.random.randn(10, 10)),
            # "base grad_norm": base_grad_norm.item(),
            f"episode-{episode}/base agent objective loss": meta_td[
                "base_agent_losses", "loss_objective"
            ].item(),
            f"episode-{episode}/base agent critic loss": meta_td[
                "base_agent_losses", "loss_critic"
            ].item(),
            f"episode-{episode}/base agent entropy loss": meta_td[
                "base_agent_losses", "loss_entropy"
            ].item(),
            f"episode-{episode}/base agent grad norm": meta_td[
                "base_agent_grad_norm"
            ].item(),
            # f"episode-{episode}/meta action": meta_td["action"].item(),
            # f"episode-{episode}/meta action prob": meta_td["probs"].item(),
            # f"episode-{episode}/meta reward": meta_td["next", "reward"].item(),
            # f"episode-{episode}/meta state [0]": meta_td["state"][0].item(),
            # f"episode-{episode}/meta state [1]": meta_td["state"][1].item(),
            f"episode-{episode}/base eval return": base_eval_td["next", "reward"]
            .sum()
            .item(),
            f"episode-{episode}/base eval state distribution": wandb.Histogram(
                base_eval_td["pos"]
            ),
            f"episode-{episode}/base eval step count": base_eval_td["step_count"]
            .max()
            .item(),
        }
    )


def print_base_rollout(td):
    # Prints visited states, actions taken, and rewards received in a base rollout
    print("<step>: (<state>, <action>, <next_reward>)")
    for i in range(td["step_count"].max().item() + 1):
        print(
            f"{i}: ({td['state'][i].item()}, {td['action'][i].item()}, {td['next', 'reward'][i].item()})"
        )
    print(f"Return: {td['next', 'reward'].sum().item()}")
    print()
