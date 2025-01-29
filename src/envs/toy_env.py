import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torchrl.data import Composite, OneHot, UnboundedContinuous, UnboundedDiscrete
from torchrl.envs import EnvBase
from torchrl.envs.transforms import (Compose, RenameTransform, StepCounter,
                                     TransformedEnv)
from torchrl.envs.utils import check_env_specs


class ToyEnv(EnvBase):

    def __init__(
        self,
        left_reward,
        right_reward,
        down_reward,
        up_reward,
        n_states,
        shortcut_steps,
        big_reward,
        punishment,
        gamma,
        constraints_active,
        random_start=False,
        seed=None,
        device="cpu",
    ):
        super().__init__(device=device, batch_size=[])

        assert (n_states-2) % shortcut_steps == 0, "n_states must be 2 more than a multiple of shortcut_steps"
        self.left_reward = left_reward
        self.right_reward = right_reward
        self.down_reward = down_reward
        self.up_reward = up_reward
        self.n_states = n_states
        self.shortcut_steps = shortcut_steps
        self.big_reward = big_reward
        self.punishment = punishment
        self.gamma = gamma
        self.random_start = random_start

        self.n_actions = 4

        self.constraints_active = constraints_active

        self._make_spec()
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    def _make_spec(self):
        self.observation_spec = Composite(
            # There does not seem to be a "BoundedDiscrete" class, so we use the unbounded one
            observation=UnboundedDiscrete(shape=(1,), dtype=torch.long),
            normal_reward=UnboundedContinuous(shape=(1,), dtype=torch.float32),
            constraint_reward=UnboundedContinuous(shape=(1,), dtype=torch.float32),
            shape=(),
        )
        self.state_spec = Composite(
            observation=UnboundedDiscrete(shape=(1,), dtype=torch.long),
            shape=(),
        )

        self.action_spec = OneHot(4, shape=(4,), dtype=torch.long)
        # The sum of normal_reward and constraint_reward
        self.reward_spec = UnboundedContinuous(shape=(1,), dtype=torch.float32)

    def _reset(self, td):
        if td is None or td.is_empty():
            batch_shape = ()
        else:
            batch_shape = td.shape

        if self.random_start:
            state_indices = torch.randint(
                0, self.n_states, shape=(*batch_shape, 1), dtype=torch.long, device=self.device
            )
        else:
            state_indices = torch.zeros((*batch_shape, 1), dtype=torch.long, device=self.device)

        state = state_indices
        # state = (
        #     F.one_hot(state_indices, num_classes=self.n_states)
        #     .to(self.device)
        # )
        # print(f"{state=}")

        out = TensorDict(
            {
                "observation": state,
                "normal_reward": torch.zeros(
                    batch_shape, dtype=torch.float32, device=self.device
                ).unsqueeze(-1),
                "constraint_reward": torch.zeros(
                    batch_shape, dtype=torch.float32, device=self.device
                ).unsqueeze(-1),
            },
            batch_size=batch_shape,
        )
        return out

    def _set_seed(self, seed):
        rng = torch.manual_seed(seed)
        self.rng = rng

    def _step(self, td):
        state = td["observation"]
        action = td["action"]  # Action order: left, right, down, up
        # state = torch.argmax(state, dim=-1)
        action = torch.argmax(action, dim=-1)

        next_state = state.clone()

        normal_reward = self.punishment * torch.ones_like(
            state, dtype=torch.float32, device=self.device
        )
        constraint_reward = 0 * torch.ones_like(
            state, dtype=torch.float32, device=self.device
        )

        left_action = 0
        right_action = 1
        down_action = 2
        up_action = 3

        # Left action
        next_state = torch.where(action == left_action, state - 1, next_state)
        constraint_reward = torch.where(
            action == left_action, 1 * self.left_reward, constraint_reward
        )
        # Right action
        next_state = torch.where(action == right_action, state + 1, next_state)
        constraint_reward = torch.where(
            action == right_action, 1 * self.right_reward, constraint_reward
        )

        # Down action
        next_state = torch.where(
            action == down_action, state - self.shortcut_steps, next_state
        )
        constraint_reward = torch.where(
            action == down_action, 1 * self.down_reward, constraint_reward
        )
        # Up action
        next_state = torch.where(
            action == up_action, state + self.shortcut_steps, next_state
        )
        constraint_reward = torch.where(
            action == up_action, 1 * self.up_reward, constraint_reward
        )

        # Ensure that we can never move past the end pos
        next_state = torch.where(
            next_state >= self.n_states, self.n_states - 1, next_state
        )

        # Ensure that we can never move before the start pos
        next_state = torch.where(next_state < 0, -next_state, next_state)

        done = torch.zeros_like(state, dtype=torch.bool, device=self.device)

        # Big reward for reaching the end pos, overriding the possible constraints
        normal_reward = torch.where(
            next_state == self.n_states - 1, self.big_reward, normal_reward
        )
        constraint_reward = torch.where(
            next_state == self.n_states - 1, 0, constraint_reward
        )
        # If we reach final pos, we're done
        done = torch.where(next_state == self.n_states - 1, 1.0, done).to(torch.bool)

        # next_state = (
        #     F.one_hot(next_state, num_classes=self.n_states)
        #     .to(self.device)
        # )

        # For some reason, rewards have to have one dimension less
        normal_reward = normal_reward.squeeze(-1)
        constraint_reward = constraint_reward.squeeze(-1)


        reward = (
            (normal_reward + constraint_reward)
            if self.constraints_active
            else normal_reward
        )

        out = TensorDict(
            {
                "observation": next_state,
                "reward": reward,
                "normal_reward": normal_reward.unsqueeze(-1),
                "constraint_reward": constraint_reward.unsqueeze(-1),
                "done": done,
            },
            td.shape,
        )
        return out

    @staticmethod
    def calculate_xy(n_states, shortcut_steps, return_x, return_y, big_reward, punishment, gamma):
        # Assuming n_pos is even, calculate x and y
        assert (n_states-2) % shortcut_steps == 0, "n_states must be 2 more than a multiple of shortcut_steps"
        nx = n_states - 2 # Number of times we need to step 'right' to reach the end, excluding the final state
        ny = (n_states - 2) // shortcut_steps # Number of times we need to step 'up' to reach the end, excluding the final state
        x = (return_x - big_reward * gamma**nx) / sum(gamma**k for k in range(0, nx)) - punishment
        y = (return_y - big_reward * gamma**ny) / sum(gamma**k for k in range(0, ny)) - punishment
        return x, y
