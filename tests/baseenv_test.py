import unittest
import torch
from torch import Tensor
import logging

from agents import fast_policy, slow_policy
from env import BaseEnv
from utils import calc_return, DictWrapper

logging.basicConfig(level=logging.DEBUG)


class TestBaseEnv(unittest.TestCase):
    def setUp(self):
        self.config = DictWrapper(
            {
                "n_states": 20,
                "return_x": 2,
                "return_y": 1,
                "big_reward": 10,
                "gamma": 0.99,
                "punishment": 0.0,
                "device": "cpu",
            }
        )
        self.x, self.y = BaseEnv.calculate_xy(
            self.config.n_states,
            self.config.return_x,
            self.config.return_y,
            self.config.big_reward,
            self.config.gamma,
        )
        self.env = BaseEnv.get_base_env(
            left_reward=self.x,
            right_reward=self.x,
            down_reward=self.y,
            up_reward=self.y,
            n_states=self.config.n_states,
            big_reward=self.config.big_reward,
            random_start=False,
            punishment=self.config.punishment,
            seed=None,
            device=self.config.device,
        ).to(self.config.device)

    def test_returns(self):
        """Check if returns are correct for both slow and fast agents with and without constraints."""
        slow_td = self.env.rollout(200, slow_policy)
        slow_return = calc_return(
            slow_td["next", "reward"].flatten(), gamma=self.config.gamma
        )
        self.assertAlmostEqual(slow_return, self.config.return_x, places=3)

        fast_td = self.env.rollout(200, fast_policy)
        fast_return = calc_return(
            fast_td["next", "reward"].flatten(), gamma=self.config.gamma
        )
        self.assertAlmostEqual(fast_return, self.config.return_y, places=3)

        # Disable constraints
        self.env.set_constraint_weight(0.0)
        slow_td = self.env.rollout(200, slow_policy)
        slow_return = calc_return(
            slow_td["next", "reward"].flatten(), gamma=self.config.gamma
        )
        self.assertAlmostEqual(
            slow_return,
            self.config.gamma ** (self.config.n_states - 2) * self.config.big_reward,
            places=3,
        )
        fast_td = self.env.rollout(200, fast_policy)
        fast_return = calc_return(
            fast_td["next", "reward"].flatten(), gamma=self.config.gamma
        )
        self.assertAlmostEqual(
            fast_return,
            self.config.gamma ** ((self.config.n_states - 2) // 2)
            * self.config.big_reward,
            places=3,
        )

    def test_true_returns(self):
        """Check if 'true_reward' reflects the true rewards for both slow and fast agents."""
        # With constraints enabled (1)
        slow_td_1 = self.env.rollout(200, slow_policy)
        slow_return_1 = calc_return(
            slow_td_1["next", "reward"].flatten(), gamma=self.config.gamma
        )
        slow_true_return_1 = calc_return(
            slow_td_1["next", "true_reward"].flatten(), gamma=self.config.gamma
        )

        fast_td_1 = self.env.rollout(200, fast_policy)
        fast_return_1 = calc_return(
            fast_td_1["next", "reward"].flatten(), gamma=self.config.gamma
        )
        fast_true_return_1 = calc_return(
            fast_td_1["next", "true_reward"].flatten(), gamma=self.config.gamma
        )

        # Without constraints enabled (0)
        self.env.set_constraint_weight(0.0)
        slow_td_0 = self.env.rollout(200, slow_policy)
        # Print rewards for debugging
        slow_true_return_0 = calc_return(
            slow_td_0["next", "true_reward"].flatten(), gamma=self.config.gamma
        )

        fast_td_0 = self.env.rollout(200, fast_policy)
        fast_true_return_0 = calc_return(
            fast_td_0["next", "true_reward"].flatten(), gamma=self.config.gamma
        )

        # The Test: The true rewards experienced without constraints should be the same
        # as the normal rewards experienced with constraints.
        self.assertAlmostEqual(slow_return_1, slow_true_return_0, places=3)
        self.assertAlmostEqual(fast_return_1, fast_true_return_0, places=3)
