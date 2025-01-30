import logging
import os
import sys
import unittest

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "../src/envs"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from src.agents.base_agents import fast_policy, slow_policy
from src.envs.toy_env import ToyEnv

from utils import calc_return

logging.basicConfig(level=logging.DEBUG)


class TestToyEnv(unittest.TestCase):
    """
    Tests for the ToyEnv environment. Behavior for each move is tested separately. Tests for the two types of returns are also included.
    """
    def setUp(self):
        self.config = {
            "n_states": 10,
            "shortcut_steps": 3,
            "return_x": 2,
            "return_y": 1,
            "big_reward": 10,
            "constraints_active": False,
            "device": "cpu",
            "max_steps": 200,
            "gamma": 0.99,
            "punishment": 0.0,
            "random_start": False,
            "device": "cpu",
        }
        x, y = ToyEnv.calculate_xy(
            n_states=self.config["n_states"],
            shortcut_steps=self.config["shortcut_steps"],
            return_x=self.config["return_x"],
            return_y=self.config["return_y"],
            big_reward=self.config["big_reward"],
            gamma=self.config["gamma"],
        )
        self.config["left_reward"] = x
        self.config["right_reward"] = x
        self.config["down_reward"] = y
        self.config["up_reward"] = y

        self.env = ToyEnv(
            left_reward=self.config["left_reward"],
            right_reward=self.config["right_reward"],
            down_reward=self.config["down_reward"],
            up_reward=self.config["up_reward"],
            n_states=self.config["n_states"],
            shortcut_steps=self.config["shortcut_steps"],
            big_reward=self.config["big_reward"],
            punishment=self.config["punishment"],
            gamma=self.config["gamma"],
            constraints_active=self.config["constraints_active"],
            device=self.config["device"],
        )

    def test_move_left(self):
        # Moving left from start position should not move us
        td = self.env.reset()
        td["action"] = torch.tensor(
            [1, 0, 0, 0]
        )
        td = self.env.step(td)
        self.assertTrue(
            torch.equal(
                td["next", "observation"],
                torch.tensor([0]),
            )
        )
        self.assertAlmostEqual(
            td["next", "normal_reward"].item(),
            0.0,
            places=3,
        )
        self.assertAlmostEqual(
            td["next", "constraint_reward"].item(),
            self.config["left_reward"],
            places=3,
        )

        # Moving left from second position should move us
        td = self.env.reset()
        td["observation"] = torch.tensor([1])
        td["action"] = torch.tensor(
            [1, 0, 0, 0]
        )
        td = self.env.step(td)
        self.assertTrue(
            torch.equal(
                td["next", "observation"],
                torch.tensor([0]),
            ),
            "Expected {}, got {}".format(torch.tensor([0]), td["next", "observation"]),
        )
        self.assertAlmostEqual(
            td["next", "normal_reward"].item(),
            0.0,
            places=3,
        )
        self.assertAlmostEqual(
            td["next", "constraint_reward"].item(),
            self.config["left_reward"],
            places=3,
        )


    def test_move_right(self):
        # Moving right from start position should move us
        td = self.env.reset()
        td["action"] = torch.tensor(
            [0, 1, 0, 0]
        )
        td = self.env.step(td)
        self.assertTrue(
            torch.equal(
                td["next", "observation"],
                torch.tensor([1]),
            )
        )
        self.assertAlmostEqual(
            td["next", "normal_reward"].item(),
            0.0,
            places=3,
        )
        self.assertAlmostEqual(
            td["next", "constraint_reward"].item(),
            self.config["right_reward"],
            places=3,
        )

        # Moving right from pre-terminal state should move us and give us the big reward
        td = self.env.reset()
        td["observation"] = torch.tensor([self.config["n_states"]-2])
        td["action"] = torch.tensor(
            [0, 1, 0, 0]
        )
        td = self.env.step(td)
        self.assertTrue(
            torch.equal(
                td["next", "observation"],
                torch.tensor([self.config["n_states"]-1]),
            )
        )
        self.assertAlmostEqual(
            td["next", "normal_reward"].item(),
            self.config["big_reward"],
            places=3,
        )
        self.assertAlmostEqual(
            td["next", "constraint_reward"].item(),
            self.config["right_reward"],
            places=3,
        )

    def test_move_down(self):
        # Moving down from the second position should not move us
        td = self.env.reset()
        td["observation"] = torch.tensor([1])
        td["action"] = torch.tensor(
            [0, 0, 1, 0]
        )
        td = self.env.step(td)
        self.assertTrue(
            torch.equal(
                td["next", "observation"],
                torch.tensor([1]),
            )
        )
        self.assertAlmostEqual(
            td["next", "normal_reward"].item(),
            0.0,
            places=3,
        )
        self.assertAlmostEqual(
            td["next", "constraint_reward"].item(),
            self.config["down_reward"],
            places=3,
        )

        # Moving down from the third position should move us to the starting position
        td = self.env.reset()
        td["observation"] = torch.tensor([2])
        td["action"] = torch.tensor(
            [0, 0, 1, 0]
        )
        td = self.env.step(td)
        self.assertTrue(
            torch.equal(
                td["next", "observation"],
                torch.tensor([0]),
            )
        )
        self.assertAlmostEqual(
            td["next", "normal_reward"].item(),
            0.0,
            places=3,
        )
        self.assertAlmostEqual(
            td["next", "constraint_reward"].item(),
            self.config["down_reward"],
            places=3,
        )

        # Moving down from the pre-terminal position should move us backwards by the shortcut steps
        td = self.env.reset()
        td["observation"] = torch.tensor([self.config["n_states"]-2])
        td["action"] = torch.tensor(
            [0, 0, 1, 0]
        )
        td = self.env.step(td)
        self.assertTrue(
            torch.equal(
                td["next", "observation"],
                torch.tensor([self.config["n_states"]-2-self.config["shortcut_steps"]]),
            )
        )
        self.assertAlmostEqual(
            td["next", "normal_reward"].item(),
            0.0,
            places=3,
        )
        self.assertAlmostEqual(
            td["next", "constraint_reward"].item(),
            self.config["down_reward"],
            places=3,
        )

    def test_move_up(self):
        # Moving up from the pre-terminal position should not move us
        td = self.env.reset()
        td["observation"] = torch.tensor([self.config["n_states"]-2])
        td["action"] = torch.tensor(
            [0, 0, 0, 1]
        )
        td = self.env.step(td)
        self.assertTrue(
            torch.equal(
                td["next", "observation"],
                torch.tensor([self.config["n_states"]-2]),
            )
        )
        self.assertAlmostEqual(
            td["next", "normal_reward"].item(),
            0.0,
            places=3,
        )
        self.assertAlmostEqual(
            td["next", "constraint_reward"].item(),
            self.config["up_reward"],
            places=3,
        )

        # Moving up from the pre-pre-terminal state should move us to the terminal state and give a big reward
        td = self.env.reset()
        td["observation"] = torch.tensor([self.config["n_states"]-3])
        td["action"] = torch.tensor(
            [0, 0, 0, 1]
        )
        td = self.env.step(td)
        self.assertTrue(
            torch.equal(
                td["next", "observation"],
                torch.tensor([self.config["n_states"]-1]),
            )
        )
        self.assertAlmostEqual(
            td["next", "normal_reward"].item(),
            self.config["big_reward"],
            places=3,
        )
        self.assertAlmostEqual(
            td["next", "constraint_reward"].item(),
            self.config["up_reward"],
            places=3,
        )

        # Moving up from the first position should move us forwards by the shortcut steps
        td = self.env.reset()
        td["observation"] = torch.tensor([0])
        td["action"] = torch.tensor(
            [0, 0, 0, 1]
        )
        td = self.env.step(td)
        self.assertTrue(
            torch.equal(
                td["next", "observation"],
                torch.tensor([self.config["shortcut_steps"]]),
            )
        )
        self.assertAlmostEqual(
            td["next", "normal_reward"].item(),
            0.0,
            places=3,
        )
        self.assertAlmostEqual(
            td["next", "constraint_reward"].item(),
            self.config["up_reward"],
            places=3,
        )

    def test_slow_return(self):
        """Check if the slow policy returns the expected return"""
        slow_rollout = self.env.rollout(1000, slow_policy)
        slow_return = calc_return((slow_rollout["next", "normal_reward"]+slow_rollout["next", "constraint_reward"]).flatten(), self.config["gamma"])
        self.assertAlmostEqual(
            slow_return,
            self.config["return_x"],
            places=3,
        )

    def test_fast_return(self):
        """Check if the fast policy returns the expected return"""
        fast_rollout = self.env.rollout(1000, fast_policy)
        fast_return = calc_return((fast_rollout["next", "normal_reward"]+fast_rollout["next", "constraint_reward"]).flatten(), self.config["gamma"])
        self.assertAlmostEqual(
            fast_return,
            self.config["return_y"],
            places=3,
        )
