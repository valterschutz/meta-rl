import logging
import os
import sys
import unittest

import torch
from torch.nn import functional as F
from torch import Tensor
from torchrl.envs.utils import step_mdp

sys.path.append(os.path.join(os.path.dirname(__file__), "../src/envs"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../src/agents"))

from toy_agents import fast_policy, slow_policy
from toy_env import ToyEnv, get_toy_env

from utils import calc_return, DictWrapper

logging.basicConfig(level=logging.DEBUG)


class TestToyEnv(unittest.TestCase):
    def setUp(self):
        self.config = {
            "n_states": 20,
            "return_x": 2,
            "return_y": 1,
            "big_reward": 10,
            "constraints_active": False,
            "device": "cpu",
        }
        self.gamma = 0.99
        self.env = get_toy_env(self.config, self.gamma)

    def test_move_horizontally(self):
        # See if moving sideways is possible and gives expected rewards
        td = self.env.reset()
        td["action"] = torch.tensor(
            [0, 1, 0, 0], device=self.config["device"]
        )  # Move right
        td = self.env.step(td)
        self.assertTrue(
            torch.equal(
                td["next", "state"].argmax(dim=-1),
                torch.tensor(1, device=self.config["device"]),
            )
        )
        self.assertAlmostEqual(
            td["next", "normal_reward"].item(),
            0.0,
            places=3,
        )
        self.assertAlmostEqual(
            td["next", "constraint_reward"].item(),
            self.env.right_reward,
            places=3,
        )
        td = step_mdp(td)

        td["action"] = torch.tensor(
            [1, 0, 0, 0], device=self.config["device"]
        )  # Move left
        td = self.env.step(td)
        self.assertTrue(
            torch.equal(
                td["next", "state"].argmax(dim=-1),
                torch.tensor(0, device=self.config["device"]),
            )
        )
        self.assertAlmostEqual(
            td["next", "normal_reward"].item(),
            0.0,
            places=3,
        )
        self.assertAlmostEqual(
            td["next", "constraint_reward"].item(),
            self.env.left_reward,
            places=3,
        )

    def test_move_vertically(self):
        # See if moving sideways is possible and gives expected rewards
        td = self.env.reset()
        td["action"] = torch.tensor(
            [0, 0, 0, 1], device=self.config["device"]
        )  # Move up
        td = self.env.step(td)
        self.assertTrue(
            torch.equal(
                td["next", "state"].argmax(dim=-1),
                torch.tensor(2, device=self.config["device"]),
            )
        )
        self.assertAlmostEqual(
            td["next", "normal_reward"].item(),
            0.0,
            places=3,
        )
        self.assertAlmostEqual(
            td["next", "constraint_reward"].item(),
            self.env.up_reward,
            places=3,
        )
        td = step_mdp(td)

        td["action"] = torch.tensor(
            [0, 0, 1, 0], device=self.config["device"]
        )  # Move down
        td = self.env.step(td)
        self.assertTrue(
            torch.equal(
                td["next", "state"].argmax(dim=-1),
                torch.tensor(0, device=self.config["device"]),
            )
        )
        self.assertAlmostEqual(
            td["next", "normal_reward"].item(),
            0.0,
            places=3,
        )
        self.assertAlmostEqual(
            td["next", "constraint_reward"].item(),
            self.env.down_reward,
            places=3,
        )

    def test_normal_reward(self):
        # See if we get the big reward if reaching goal
        td = self.env.reset()
        td["state"] = F.one_hot(
            torch.tensor([self.config["n_states"] - 2], device=self.config["device"]),
            num_classes=self.config["n_states"],
        ).to(torch.float32)
        td["action"] = torch.tensor(
            [0, 1, 0, 0], device=self.config["device"]
        )  # Move right
        td = self.env.step(td)
        self.assertTrue(
            td["next", "state"].argmax(dim=-1).item() == self.config["n_states"] - 1,
            msg=f"{td['next', 'state'].argmax(dim=-1).item()=} instead of {self.config['n_states']-1=}",
        )
        self.assertAlmostEqual(
            td["next", "normal_reward"].item(),
            self.env.big_reward,
            places=3,
        )
        # No constraints at the end
        self.assertAlmostEqual(
            td["next", "constraint_reward"].item(),
            0,
            places=3,
        )

    def test_returns(self):
        """Check if returns are correct for both slow and fast agents with and without constraints."""

        # Enable constraints
        self.env.constraints_active = True
        slow_td = self.env.rollout(200, slow_policy)
        slow_return = calc_return(slow_td["next", "reward"].flatten(), gamma=self.gamma)
        self.assertAlmostEqual(slow_return, self.config["return_x"], places=3)

        fast_td = self.env.rollout(200, fast_policy)
        fast_return = calc_return(fast_td["next", "reward"].flatten(), gamma=self.gamma)
        self.assertAlmostEqual(fast_return, self.config["return_y"], places=3)

        # Disable constraints
        self.env.constraints_active = False
        slow_td = self.env.rollout(200, slow_policy)
        slow_return = calc_return(slow_td["next", "reward"].flatten(), gamma=self.gamma)
        self.assertAlmostEqual(
            slow_return,
            self.gamma ** (self.config["n_states"] - 2) * self.config["big_reward"],
            places=3,
        )
        fast_td = self.env.rollout(200, fast_policy)
        fast_return = calc_return(fast_td["next", "reward"].flatten(), gamma=self.gamma)
        self.assertAlmostEqual(
            fast_return,
            self.gamma ** ((self.config["n_states"] - 2) // 2)
            * self.config["big_reward"],
            places=3,
        )
