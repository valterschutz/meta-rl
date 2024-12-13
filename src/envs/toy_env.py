from torchrl.envs import EnvBase
import torch
from torchrl.data import UnboundedContinuous, Composite, OneHot
from tensordict import TensorDict
from torchrl.envs.transforms import (
    TransformedEnv,
    Compose,
    StepCounter,
    RenameTransform,
)
from torchrl.envs.utils import check_env_specs
import torch.nn.functional as F


class ToyEnv(EnvBase):
    batch_locked = False

    def __init__(
        self,
        left_reward,
        right_reward,
        down_reward,
        up_reward,
        n_states,
        big_reward,
        random_start=False,
        # punishment=0,
        seed=None,
        device="cpu",
        # left_weight=1.0,
        # right_weight=1.0,
        # down_weight=1.0,
        # up_weight=1.0,
    ):
        super().__init__(device=device, batch_size=[])

        assert n_states % 2 == 0, "n_states only tested for even numbers"
        self.left_reward = left_reward
        self.right_reward = right_reward
        self.down_reward = down_reward
        self.up_reward = up_reward
        self.n_states = n_states
        self.big_reward = big_reward
        self.random_start = random_start
        # self.punishment = punishment
        # self.left_weight = left_weight
        # self.right_weight = right_weight
        # self.down_weight = down_weight
        # self.up_weight = up_weight

        self._make_spec()
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    def _make_spec(self):
        self.observation_spec = Composite(
            state=OneHot(self.n_states, shape=(self.n_states,), dtype=torch.float32),
            constraint_reward=UnboundedContinuous(shape=(1), dtype=torch.float32),
            shape=(),
        )
        self.state_spec = Composite(
            state=OneHot(self.n_states, shape=(self.n_states,), dtype=torch.float32),
            shape=(),
        )

        self.action_spec = OneHot(4, shape=(4,), dtype=torch.float32)
        # The normal, sparse reward
        self.reward_spec = UnboundedContinuous(shape=(1,), dtype=torch.float32)

    def _reset(self, td):
        if td is None or td.is_empty():
            shape = ()
        else:
            shape = td.shape

        if self.random_start:
            state_indices = torch.randint(
                0, self.n_states, shape=shape, dtype=torch.long, device=self.device
            )
        else:
            state_indices = torch.zeros(shape, dtype=torch.long, device=self.device)

        state = (
            F.one_hot(state_indices, num_classes=self.n_states)
            .to(torch.float32)
            .to(self.device)
        )
        # print(f"{state=}")

        out = TensorDict(
            {
                "state": state,
                "constraint_reward": torch.zeros(
                    shape, dtype=torch.float32, device=self.device
                ).unsqueeze(-1),
            },
            batch_size=shape,
        )
        return out

    def _set_seed(self, seed):
        rng = torch.manual_seed(seed)
        self.rng = rng

    def _step(self, td):
        state = td["state"]
        action = td["action"]  # Action order: left, right, down, up
        # x, y, n_pos, big_reward = self.x, self.y, self.n_pos, self.big_reward
        state = torch.argmax(state, dim=-1)
        action = torch.argmax(action, dim=-1)

        next_state = state.clone()
        reward = 0 * torch.ones_like(state, dtype=torch.int, device=self.device)
        constraint_reward = 0 * torch.ones_like(
            state, dtype=torch.int, device=self.device
        )

        mask_even = state % 2 == 0

        # Enable left action by default
        next_state = torch.where(action == 0, state - 1, next_state)
        # Enable right action by default
        next_state = torch.where(action == 1, state + 1, next_state)

        # For even pos, enable down and up actions
        # Down action
        next_state = torch.where(mask_even & (action == 2), state - 2, next_state)
        # Up action
        next_state = torch.where(mask_even & (action == 3), state + 2, next_state)

        # Left action
        constraint_reward = torch.where(
            action == 0, 1 * self.left_reward, constraint_reward
        )
        # Right action
        constraint_reward = torch.where(
            action == 1, 1 * self.right_reward, constraint_reward
        )

        # Down action
        constraint_reward = torch.where(
            mask_even & (action == 2), 1 * self.down_reward, constraint_reward
        )
        # Up action
        constraint_reward = torch.where(
            mask_even & (action == 3), 1 * self.up_reward, constraint_reward
        )

        # Ensure that we can never move past the end pos
        next_state = torch.where(
            next_state >= self.n_states, self.n_states - 1, next_state
        )

        # Ensure that we can never move before the start pos
        next_state = torch.where(next_state < 0, state, next_state)

        done = torch.zeros_like(state, dtype=torch.bool, device=self.device)
        # constraint_reward = torch.where(
        #     next_state == state, -self.punishment, constraint_reward
        # )

        # Big reward for reaching the end pos, overriding the possible constraints
        sparse_reward = torch.where(
            next_state == self.n_states - 1, self.big_reward, reward
        )
        # If we reach final pos, we're done
        done = torch.where(next_state == self.n_states - 1, 1.0, done).to(torch.bool)

        next_state = (
            F.one_hot(next_state, num_classes=self.n_states)
            .to(torch.float32)
            .to(self.device)
        )

        out = TensorDict(
            {
                "state": next_state,
                "reward": sparse_reward,
                "constraint_reward": constraint_reward.unsqueeze(-1),
                "done": done,
            },
            td.shape,
        )
        return out

    @staticmethod
    def calculate_xy(n_states, return_x, return_y, big_reward, gamma):
        # Assuming n_pos is even, calculate x and y
        assert n_states % 2 == 0
        nx = n_states - 2
        ny = (n_states - 2) // 2
        x = (return_x - big_reward * gamma**nx) / sum(gamma**k for k in range(0, nx))
        y = (return_y - big_reward * gamma**ny) / sum(gamma**k for k in range(0, ny))
        return x, y

    def set_left_weight(self, left_weight):
        self.left_weight = left_weight

    def set_right_weight(self, right_weight):
        self.right_weight = right_weight

    def set_down_weight(self, down_weight):
        self.down_weight = down_weight

    def set_up_weight(self, up_weight):
        self.up_weight = up_weight

    def set_constraint_weight(self, weight):
        # Clip weight to be between 0 and 1
        weight = max(0, min(weight, 1))
        self.set_left_weight(weight)
        self.set_right_weight(weight)
        self.set_down_weight(weight)
        self.set_up_weight(weight)


def get_toy_env(env_config, gamma):
    x, y = ToyEnv.calculate_xy(
        env_config["n_states"],
        env_config["return_x"],
        env_config["return_y"],
        env_config["big_reward"],
        gamma,
    )
    env = ToyEnv(
        left_reward=x,
        right_reward=x,
        down_reward=y,
        up_reward=y,
        n_states=env_config["n_states"],
        big_reward=env_config["big_reward"],
        random_start=False,
        # punishment=env_config["punishment"],
        seed=None,
        device=env_config["device"],
    ).to(env_config["device"])

    # Rename reward to normal_reward
    env = TransformedEnv(
        env,
        Compose(
            StepCounter(),
            RenameTransform(
                in_keys=["reward"], out_keys=["normal_reward"], create_copy=True
            ),
        ),
    )

    check_env_specs(env)
    return env
