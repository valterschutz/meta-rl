import unittest
import torch
from torch import Tensor

from agents import fast_policy, slow_policy
from env import BaseEnv
from utils import calc_return, DictWrapper


class TestBaseEnv(unittest.TestCase):
    def test_true_rewards_slow_agent(self):
        """See if the key 'true_reward' reflects the true rewards for the slow agent."""
        config = DictWrapper(
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
        x, y = BaseEnv.calculate_xy(
            config.n_states,
            config.return_x,
            config.return_y,
            config.big_reward,
            config.gamma,
        )
        # Base env
        env = BaseEnv.get_base_env(
            left_reward=x,
            right_reward=x,
            down_reward=y,
            up_reward=y,
            n_states=config.n_states,
            big_reward=config.big_reward,
            random_start=False,
            punishment=config.punishment,
            seed=None,
            device=config.device,
        ).to(config.device)
        # With constraints enabled (1)
        slow_td_1 = env.rollout(200, slow_policy)
        slow_return_1 = calc_return(
            slow_td_1["next", "reward"].flatten(), gamma=config.gamma
        )
        slow_true_return_1 = calc_return(
            slow_td_1["true_reward"].flatten(), gamma=config.gamma
        )
        # Without constraints enabled (0)
        env.set_constraint_weight(0.0)
        slow_td_0 = env.rollout(200, slow_policy)
        slow_return_0 = calc_return(
            slow_td_1["next", "reward"].flatten(), gamma=config.gamma
        )
        slow_true_return_0 = calc_return(
            slow_td_1["true_reward"].flatten(), gamma=config.gamma
        )

        # The Test: The true rewards experienced without constraints should be the same
        # as the normal rewards experienced with constraints.
        self.assertAlmostEqual(slow_return_1, slow_true_return_0, places=6)

    def test_true_rewards_fast_agent(self):
        """See if the key 'true_reward' reflects the true rewards for the slow agent."""
        config = DictWrapper(
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
        x, y = BaseEnv.calculate_xy(
            config.n_states,
            config.return_x,
            config.return_y,
            config.big_reward,
            config.gamma,
        )
        # Base env
        env = BaseEnv.get_base_env(
            left_reward=x,
            right_reward=x,
            down_reward=y,
            up_reward=y,
            n_states=config.n_states,
            big_reward=config.big_reward,
            random_start=False,
            punishment=config.punishment,
            seed=None,
            device=config.device,
        ).to(config.device)
        # With constraints enabled (1)
        fast_td_1 = env.rollout(200, fast_policy)
        fast_return_1 = calc_return(
            fast_td_1["next", "reward"].flatten(), gamma=config.gamma
        )
        fast_true_return_1 = calc_return(
            fast_td_1["true_reward"].flatten(), gamma=config.gamma
        )
        # Without constraints enabled (0)
        env.set_constraint_weight(0.0)
        fast_td_0 = env.rollout(200, fast_policy)
        fast_return_0 = calc_return(
            fast_td_1["next", "reward"].flatten(), gamma=config.gamma
        )
        fast_true_return_0 = calc_return(
            fast_td_1["true_reward"].flatten(), gamma=config.gamma
        )

        # The Test: The true rewards experienced without constraints should be the same
        # as the normal rewards experienced with constraints.
        self.assertAlmostEqual(fast_return_1, fast_true_return_0, places=6)
