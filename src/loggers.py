from collections import defaultdict
import wandb
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
from utils import calc_return
from typing import Dict, List, Any, Protocol
from tensordict import TensorDict
from tensordict.tensordict import TensorDictBase

class Logger(Protocol):
    """
    Interface for loggers, used by OffpolicyTrainer
    """

    def train_log(self, td: TensorDictBase) -> None:
        """
        Called once every training batch.
        """
        ...

    def eval_log(self, td: TensorDictBase) -> None:
        """
        Called every now and then, when the agent is evaluated.
        """
        ...

    def dump(self) -> Dict[str, List[Any]]:
        """
        Should return a dictionary with lists of values that describe the training run.
        """
        ...

class ToyTabularQLogger(Logger):
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.batch_count = 0

        self.history = defaultdict(list)

        self.optimal_non_constrained_qvalues = self.env.calc_optimal_qvalues(constraints_active=False)
        self.optimal_constrained_qvalues = self.env.calc_optimal_qvalues(constraints_active=True)

        self.optimal_non_constrained_policy = self.env.calc_optimal_policy(constraints_active=False)
        self.optimal_constrained_policy = self.env.calc_optimal_policy(constraints_active=True)


    def train_log(self, td):
        self.batch_count += 1
        # self.history["qvalues"].append(self.agent.qvalues.clone())
        # self.history["epsilon"].append(self.agent.epsilon)
        constrained_qvalue_offset = (self.agent.qvalues[:-1] - self.optimal_constrained_qvalues[:-1]).mean().item()
        non_constrained_qvalue_offset = (self.agent.qvalues[:-1] - self.optimal_non_constrained_qvalues[:-1]).mean().item()
        self.history["constrained qvalue offset"].append(constrained_qvalue_offset)
        self.history["non-constrained qvalue offset"].append(non_constrained_qvalue_offset)

        policy = self.agent.qvalues[:-1].argmax(dim=-1)

        constrained_policy_offset = (policy - self.optimal_constrained_policy).abs().sum().item()
        non_constrained_policy_offset = (policy - self.optimal_non_constrained_policy).abs().sum().item()
        self.history["constrained policy offset"].append(constrained_policy_offset)
        self.history["non-constrained policy offset"].append(non_constrained_policy_offset)


        # wandb.log({
        #     "batch": self.batch_count,
        #     "qvalue": self.agent.qvalues[:-1].mean().item(),
        #     "non-constrained qvalue offset": non_constrained_qvalue_offset,
        #     "constrained qvalue offset": constrained_qvalue_offset,
        #     "non-constrained policy offset": non_constrained_policy_offset,
        #     "constrained policy offset": constrained_policy_offset,
        #     "state distribution": wandb.Histogram(td["next", "observation"].cpu()),
        #     "action distribution": wandb.Histogram(td["action"].argmax(dim=-1).cpu()),
        #     # "preferred actions according to qvalues": wandb.Histogram(self.agent.qvalues[:-1].argmax(dim=-1).cpu()),
        #     "td error": self.agent.latest_td_errors.mean().item(),
        #     "normal reward": td["next", "normal_reward"].mean().item(),
        #     "constraint reward": td["next", "constraint_reward"].mean().item(),
        #     "reward": td["next", "reward"].mean().item(),
        #     # "distance to slow policy": self.get_distance_to_slow_policy(),
        #     # "distance to fast policy": self.get_distance_to_fast_policy(),
        #     # "policy matrix": wandb.Image(self.get_policy_matrix(), "Policy matrix"),
        #     # "qvalues": wandb.Image(self.get_qvalues(), "Q values")
        #     # "policy matrix and qvalues": wandb.Image(fig, "Policy matrix and Q values")
        # })

    def eval_log(self, td):
        # fig, axs = plt.subplots(1, 3)
        # fig.subplots_adjust(wspace=0.5)
        # p0 = axs[0].imshow(self.agent.qvalues[:-1])
        # axs[0].set_yticks(np.arange(self.agent.qvalues.shape[0]-1), self.agent.qvalues[:-1].argmax(dim=-1).numpy())
        # axs[0].set_title("Agent Q values")
        # fig.colorbar(p0, ax=axs[0])
        # p1 = axs[1].imshow(self.optimal_qvalues[:-1])
        # axs[1].set_yticks(np.arange(self.optimal_qvalues.shape[0]-1), self.optimal_qvalues[:-1].argmax(dim=-1).numpy())
        # axs[1].set_title("Optimal Q values")
        # fig.colorbar(p1, ax=axs[1])
        # p2 = axs[2].imshow(self.agent.qvalues[:-1]-self.optimal_qvalues[:-1])
        # axs[2].set_title("Q value offset")
        # fig.colorbar(p2, ax=axs[2])

        # eval_normal_return = calc_return((td["next", "normal_reward"]).flatten(), self.env.gamma)
        # eval_true_return = calc_return((td["next", "normal_reward"]+td["next","constraint_reward"]).flatten(), self.env.gamma)
        # wandb.log({
            # "batch": self.batch_count,
            # "eval normal return": eval_normal_return,
            # "eval true return": eval_true_return,
        # })
        # wandb.log({
        #     "qvalues": wandb.Image(fig),
        # })
        pass

    def dump(self):
        return self.history
