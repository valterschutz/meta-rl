import logging
import os
import sys
import unittest

import torch
import torch.nn.functional as F

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
            "batch_size": (1,),
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
            batch_size=self.config["batch_size"],
            shortcut_steps=self.config["shortcut_steps"],
            big_reward=self.config["big_reward"],
            punishment=self.config["punishment"],
            gamma=self.config["gamma"],
            constraints_active=self.config["constraints_active"],
            device=self.config["device"],
        )

    def assert_helper(self, starting_state, action, ending_state, normal_reward, constraint_reward):
        """
        Asserts that starting in the starting_state and taking the action results in the ending_state, normal_reward, and constraint_reward.
        """
        td = self.env.reset()
        td["observation"] = torch.tensor([starting_state]).unsqueeze(0)
        td["action"] = F.one_hot(torch.tensor([action]), num_classes=4).unsqueeze(0)
        td = self.env.step(td)
        self.assertEqual(
            td["next", "observation"].item(),
            ending_state
        )
        self.assertAlmostEqual(
            td["next", "normal_reward"].item(),
            normal_reward,
            places=3,
        )
        self.assertAlmostEqual(
            td["next", "constraint_reward"].item(),
            constraint_reward,
            places=3,
        )

    def test_move_left(self):
        # Moving left from start position should not move us
        self.assert_helper(
            starting_state=0,
            action=0,
            ending_state=0,
            normal_reward=0.0,
            constraint_reward=self.config["left_reward"],
        )

        # Moving left from second position should move us
        self.assert_helper(
            starting_state=1,
            action=0,
            ending_state=0,
            normal_reward=0.0,
            constraint_reward=self.config["left_reward"],
        )


    def test_move_right(self):
        # Moving right from start position should move us
        self.assert_helper(
            starting_state=0,
            action=1,
            ending_state=1,
            normal_reward=0.0,
            constraint_reward=self.config["right_reward"],
        )

        # Moving right from pre-terminal state should move us and give us the big reward
        self.assert_helper(
            starting_state=8,
            action=1,
            ending_state=9,
            normal_reward=10.0,
            constraint_reward=self.config["right_reward"],
        )

    def test_move_down(self):
        # Moving down from state 4 should not move us
        self.assert_helper(
            starting_state=4,
            action=2,
            ending_state=4,
            normal_reward=0.0,
            constraint_reward=self.config["down_reward"],
        )

        # Moving down from state 5 should take us to state 3
        self.assert_helper(
            starting_state=5,
            action=2,
            ending_state=3,
            normal_reward=0.0,
            constraint_reward=self.config["down_reward"],
        )

        # Moving down from state 6 should take us to state 3
        self.assert_helper(
            starting_state=6,
            action=2,
            ending_state=3,
            normal_reward=0.0,
            constraint_reward=self.config["down_reward"],
        )

    def test_move_up(self):
        # Moving up from state 5 should not move us
        self.assert_helper(
            starting_state=5,
            action=3,
            ending_state=5,
            normal_reward=0.0,
            constraint_reward=self.config["up_reward"],
        )

        # Moving up from state 4 should move us to state 6
        self.assert_helper(
            starting_state=4,
            action=3,
            ending_state=6,
            normal_reward=0.0,
            constraint_reward=self.config["up_reward"],
        )

        # Moving up from state 3 should take us to state 6
        self.assert_helper(
            starting_state=3,
            action=3,
            ending_state=6,
            normal_reward=0.0,
            constraint_reward=self.config["up_reward"],
        )

        # Moving up from state 8 should not move us
        self.assert_helper(
            starting_state=8,
            action=3,
            ending_state=8,
            normal_reward=0.0,
            constraint_reward=self.config["up_reward"],
        )

        # Moving up from state 7 should move us and give a big reward
        self.assert_helper(
            starting_state=7,
            action=3,
            ending_state=9,
            normal_reward=10.0,
            constraint_reward=self.config["up_reward"],
        )

    def test_slow_return(self):
        """Check if the slow policy returns the expected return"""
        slow_rollout = self.env.rollout(self.config["max_steps"], slow_policy)
        slow_return = calc_return((slow_rollout["next", "normal_reward"]+slow_rollout["next", "constraint_reward"]).flatten(), self.config["gamma"])
        self.assertAlmostEqual(
            slow_return,
            self.config["return_x"],
            places=3,
        )

    def test_fast_return(self):
        """Check if the fast policy returns the expected return"""
        fast_rollout = self.env.rollout(self.config["max_steps"], fast_policy)
        fast_return = calc_return((fast_rollout["next", "normal_reward"]+fast_rollout["next", "constraint_reward"]).flatten(), self.config["gamma"])
        self.assertAlmostEqual(
            fast_return,
            self.config["return_y"],
            places=3,
        )
