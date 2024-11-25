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


def log(pbar, meta_td):
    pbar.update(meta_td.numel())
    wandb.log(
        {
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
            # "meta action": meta_action,
            # "meta action prob": meta_action_prob.item(),
            "meta reward": meta_td["next", "reward"].item(),
        }
    )
