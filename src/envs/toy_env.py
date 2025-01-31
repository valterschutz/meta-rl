import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torchrl.data import Composite, OneHot, UnboundedContinuous, UnboundedDiscrete
from torchrl.envs import EnvBase
from torchrl.envs.transforms import (Compose, RenameTransform, StepCounter,
                                     TransformedEnv)
from torchrl.envs.utils import check_env_specs


class ToyEnv(EnvBase):
    batch_locked = False

    def __init__(
        self,
        n_states,
        shortcut_steps,
        big_reward,
        punishment, # How much to punish every action, used to encourage the agent to take the shortest path
        gamma,
        random_start=False,
        seed=None,
        device="cpu",
        left_reward=None,
        right_reward=None,
        down_reward=None,
        up_reward=None,
        return_x=None,
        return_y=None
    ):
        super().__init__(device=device, batch_size=())

        assert (n_states-1) % shortcut_steps == 0, "n_states must be 1 more than a multiple of shortcut_steps"

        self.n_states = n_states
        self.shortcut_steps = shortcut_steps
        self.big_reward = big_reward
        self.punishment = punishment
        self.gamma = gamma
        self.random_start = random_start

        self.n_actions = 4

        self._reward_init(left_reward, right_reward, down_reward, up_reward, return_x, return_y)

        self._make_spec()

        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    def _reward_init(self, left_reward, right_reward, down_reward, up_reward, return_x, return_y):
        """
        Initialize the rewards for the environment. Either supply explicit rewards or calculate them from return_x and return_y.
        """
        # If return_x is provided, use it to calculate left_reward and right_reward
        if return_x is not None:
            assert left_reward is None and right_reward is None, "Either return_x or left_reward and right_reward must be provided"
            x = self.calculate_x(self.n_states, self.shortcut_steps, return_x, self.big_reward, self.gamma)
            left_reward = x
            right_reward = x
        else:
            assert left_reward is not None and right_reward is not None, "Either return_x or left_reward and right_reward must be provided"
        # If return_y is provided, use it to calculate down_reward and up_reward
        if return_y is not None:
            assert down_reward is None and up_reward is None, "Either return_y or down_reward and up_reward must be provided"
            y = self.calculate_y(self.n_states, self.shortcut_steps, return_y, self.big_reward, self.gamma)
            down_reward = y
            up_reward = y
        else:
            assert down_reward is not None and up_reward is not None, "Either return_y or down_reward and up_reward must be provided"

        self.left_reward = left_reward
        self.right_reward = right_reward
        self.down_reward = down_reward
        self.up_reward = up_reward

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
            batch_size = ()
        else:
            batch_size = td.shape

        if self.random_start:
            state_indices = torch.randint(
                0, self.n_states, shape=(*batch_size, 1), dtype=torch.long, device=self.device
            )
        else:
            state_indices = torch.zeros((*batch_size, 1), dtype=torch.long, device=self.device)

        state = state_indices

        out = TensorDict(
            {
                "observation": state,
                "normal_reward": torch.zeros(
                    (*batch_size,1), dtype=torch.float32, device=self.device
                ),
                "constraint_reward": torch.zeros(
                    (*batch_size, 1), dtype=torch.float32, device=self.device
                )
            },
            batch_size=batch_size,
        )
        return out

    def _set_seed(self, seed):
        rng = torch.manual_seed(seed)
        self.rng = rng

    def _step(self, td):
        state = td["observation"].squeeze(-1) # Squeeze last dimension to simplify tensor calculations
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

        # Create masks for each action
        masks = {
            "left": action == left_action,
            "right": action == right_action,
            "down": action == down_action,
            "up": action == up_action,
        }

        # Compute state changes and rewards for each action
        state_changes = {
            "left": -1,
            "right": 1,
            # Trying to move down one state after a checkpoint will not move us
            # Otherwise move to the previous checkpoint
            "down": torch.where(
                ((state-1) % self.shortcut_steps) == 0,
                0,
                torch.where(
                    (state % self.shortcut_steps) == 0,
                    -self.shortcut_steps,
                    -(state % self.shortcut_steps),
                )
            ),
            # Trying to move up one state before a checkpoint will not move us
            # Otherwise move to the next checkpoint
            "up": torch.where(
                ((state+1) % self.shortcut_steps) == 0,
                0,
                self.shortcut_steps - (state % self.shortcut_steps),
            )
        }

        constraint_rewards = {
            "left": self.left_reward,
            "right": self.right_reward,
            "down": self.down_reward,
            "up": self.up_reward,
        }

        # Apply changes based on action masks
        for action_name in ["left", "right", "down", "up"]:
            mask = masks[action_name]
            next_state = torch.where(
                mask,
                torch.clip(next_state + state_changes[action_name], 0, self.n_states - 1),
                next_state,
            )
            constraint_reward = torch.where(
                mask,
                constraint_rewards[action_name] * torch.ones_like(constraint_reward),
                constraint_reward,
            )

        done = torch.zeros_like(state, dtype=torch.bool, device=self.device)

        # Big reward for reaching the end pos, additive with normal reward
        normal_reward = torch.where(
            next_state == self.n_states - 1, self.big_reward, normal_reward
        )
        done = torch.where(next_state == self.n_states - 1, True, done)

        reward = (
            (normal_reward + constraint_reward)
        )

        out = TensorDict(
            {
                "observation": next_state.unsqueeze(-1),
                "reward": reward.unsqueeze(-1),
                "normal_reward": normal_reward.unsqueeze(-1),
                "constraint_reward": constraint_reward.unsqueeze(-1),
                "done": done,
            },
            td.shape,
        )
        return out

    @staticmethod
    def calculate_x(n_states, shortcut_steps, return_x, big_reward, gamma):
        assert (n_states-1) % shortcut_steps == 0, "n_states must be 1 more than a multiple of shortcut_steps"
        nx = n_states - 1 # Number of times we need to step 'right' to reach the end
        x = (return_x - big_reward * gamma**(nx-1)) / sum(gamma**k for k in range(0, nx))
        return x

    @staticmethod
    def calculate_y(n_states, shortcut_steps, return_y, big_reward, gamma):
        assert (n_states-1) % shortcut_steps == 0, "n_states must be 1 more than a multiple of shortcut_steps"
        ny = (n_states - 1) // shortcut_steps # Number of times we need to step 'up' to reach the end
        y = (return_y - big_reward * gamma**(ny-1)) / sum(gamma**k for k in range(0, ny))
        return y

    def calc_optimal_qvalues(self, constraints_active, tol=1e-6):
        qvalues = torch.zeros(self.n_states, self.n_actions)
        states = torch.arange(self.n_states-1).repeat(self.n_actions)
        actions = torch.arange(self.n_actions).repeat_interleave(self.n_states-1)
        td = TensorDict({
            "observation": states.unsqueeze(-1),
            "action": F.one_hot(actions, num_classes=self.n_actions),
            }, batch_size=((self.n_states-1)*self.n_actions,)
        )
        td = self.step(td)
        if constraints_active:
            rewards = td["next","normal_reward"] + td["next", "constraint_reward"]
        else:
            rewards = td["next", "normal_reward"]
        delta = 1
        while delta > tol:
            old_Q = qvalues.clone()
            qvalues[states, actions] = rewards.squeeze(-1) + self.gamma * qvalues[td["next", "observation"].squeeze(-1), :].max(dim=-1).values
            delta = (qvalues - old_Q).abs().max().item()
        return qvalues

    def calc_optimal_policy(self, constraints_active, tol=1e-6):
        qvalues = self.calc_optimal_qvalues(constraints_active, tol)
        return qvalues[:-1].argmax(dim=-1)

    def get_env(env_config):
        """
        Factory method for creating a ToyEnv instance. Wraps the environment in a TransformedEnv instance.
        """

        env_config = env_config.copy() # Copy the config to avoid modifying the original

        max_steps = env_config.pop("max_steps")

        env = ToyEnv(
            **env_config
        )

        env = TransformedEnv(
            env,
            Compose(
                StepCounter(max_steps=max_steps),
            )
        )

        return env
