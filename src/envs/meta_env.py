import torch.nn as nn
import torch
from torchrl.envs import EnvBase
from tensordict import TensorDict
from itertools import peekable
import torch.nn.functional as F
from torchrl.envs.transforms import Composite, OneHot, UnboundedContinuous, Bounded
from torchrl.data import Unbounded

class MetaEnv(EnvBase):
    def __init__(self, base_env, base_agent, base_collector_fn, base_loss_keys, device, seed=None):
        super().__init__(device=device, batch_size=[])
        self.base_env = base_env
        self.base_agent = base_agent

        self.base_collector_fn = base_collector_fn
        self.base_loss_keys = base_loss_keys

        # Calculate batch size, necessary to know size of observations for meta agent
        i = iter(self.base_collector_fn())
        dummy_td = next(i)
        self.base_batch_size = dummy_td.batch_size.numel()

        self._make_spec()
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    def _make_spec(self):
        self.observation_spec = Composite(
            # The state
            base_mean_normal_reward=Unbounded(shape=(1,), dtype=torch.float32),
            base_mean_true_reward=Unbounded(shape=(1,), dtype=torch.float32),
            last_action=Bounded(low=0, high=1, shape=(1,), dtype=torch.float32),
            # Base agent that we observe
            base=Composite(
                losses=Composite(
                    **{"loss_key": Unbounded(shape=(), dtype=torch.float32) for loss_key in self.base_loss_keys},
                    batch_size=(),
                ),
                states=OneHot(
                    self.base_env.n_states,
                    shape=(self.base_batch_size, self.base_env.n_states),
                    dtype=torch.float32,
                ),
                normal_rewards=Unbounded(shape=(self.base_batch_size, 1), dtype=torch.float32),
                constraint_rewards=Unbounded(
                    shape=(self.base_batch_size, 1), dtype=torch.float32
                ),
                grad_norm=Unbounded(shape=(), dtype=torch.float32),
                batch_size=(),
            ),
            step=Unbounded(shape=(1,), dtype=torch.float32),
            shape=(),
        )
        self.state_spec = Composite(
            base_mean_normal_reward=Unbounded(shape=(1,), dtype=torch.float32),
            base_mean_constraint_reward=Unbounded(shape=(1,), dtype=torch.float32),
            last_action=Bounded(low=0, high=1, shape=(1,), dtype=torch.float32),
            step=Unbounded(shape=(1,), dtype=torch.float32),
        )
        self.action_spec = Bounded(low=0, high=1, shape=(1,), dtype=torch.float32)
        self.reward_spec = UnboundedContinuous(shape=(1,), dtype=torch.float32)

    def _reset(self, td):
        agent = BaseAgent(self.base_env, self.device, buffer_size, min_buffer_size, batch_size, sub_batch_size, alpha, beta, num_epochs)

        collector = SyncDataCollector(
            env,
            agent.policy_module,
            frames_per_batch=batch_size,
            total_frames=total_frames,
            split_trajs=False,
            device=device,
        )

        return TensorDict(
            {
                "base_mean_normal_reward": torch.tensor([0.0], dtype=torch.float32),
                "base_mean_constraint_reward": torch.tensor([0.0], dtype=torch.float32),
                "last_action": torch.tensor([0], dtype=torch.float32),
                "base": TensorDict(
                    {
                        "losses": TensorDict(
                            {
                                loss_key: torch.tensor(
                                    0.0, device=self.device, dtype=torch.float32
                                ) for loss_key in self.loss_keys
                            },
                            batch_size=(),
                        ),
                        "grad_norm": torch.tensor(0.0),
                        "states": F.one_hot(
                            torch.zeros(self.base_batch_size, dtype=torch.long),
                            num_classes=self.base_env.n_states,
                        ).to(torch.float32),
                        "normal_rewards": torch.zeros(
                            (self.base_batch_size, 1), dtype=torch.float32
                        ),
                        "constraint_rewards": torch.zeros(
                            (self.base_batch_size, 1), dtype=torch.float32
                        ),
                    },
                    batch_size=(),
                ),
                "step": torch.tensor([0.0], dtype=torch.float32),
            },
            batch_size=(),
        )

    def _set_seed(self, seed):
        rng = torch.manual_seed(seed)
        self.rng = rng

    def _step(self, meta_td, verbose=False):
        """Takes one step in the meta environment, returning the next meta td and also the base td"""

        # Apply meta action, which will affect self.base_iter
        self.base_env.set_constraint_weight(meta_td["action"].item())

        next_meta_td = TensorDict()

        # Get next base batch and update base agent
        base_td = next(self.base_iter)
        next_meta_td["base_true_mean_reward"] = base_td["next", "true_reward"].mean(0)
        next_meta_td["base_mean_reward"] = base_td["next", "reward"].mean(0)
        next_meta_td["base_std_reward"] = base_td["next", "reward"].std(0)
        next_meta_td["last_action"] = meta_td["action"].detach()  # Note detach
        next_meta_td["reward"] = base_td["next", "true_reward"].mean(0)
        base_losses, base_grad_norm = self.base_agent.process_batch(base_td)
        next_meta_td["done"] = not self.is_batches_remaining(self.base_iter)
        next_meta_td["step"] = meta_td["step"] + 1
        next_meta_td["constant"] = meta_td["constant"]

        next_meta_td["base", "states"] = base_td["state"]
        next_meta_td["base", "rewards"] = base_td["next", "reward"]
        next_meta_td["base", "true_rewards"] = base_td["next", "true_reward"]
        next_meta_td["base", "losses"] = base_losses
        next_meta_td["base", "grad_norm"] = base_grad_norm

        return next_meta_td

    @staticmethod
    def is_batches_remaining(peekable_iterator):
        try:
            peekable_iterator.peek()
            return True
        except StopIteration:
            return False
