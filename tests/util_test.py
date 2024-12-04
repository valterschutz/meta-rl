import unittest
import torch
from torch import Tensor

from utils import calc_return


class TestCalcReturn(unittest.TestCase):
    def test_single_reward(self):
        """Test a single reward tensor."""
        rewards = torch.tensor([1.0])
        gamma = 0.99
        expected_return = 1.0
        self.assertAlmostEqual(calc_return(rewards, gamma), expected_return, places=6)

    def test_multiple_rewards(self):
        """Test multiple rewards with discounting."""
        rewards = torch.tensor([1.0, 0.5, 0.25])
        gamma = 0.9
        expected_return = 1.0 + 0.5 * 0.9 + 0.25 * (0.9**2)
        self.assertAlmostEqual(calc_return(rewards, gamma), expected_return, places=6)

    def test_discount_start(self):
        """Test with non-zero discount start."""
        rewards = torch.tensor([1.0, 0.5])
        gamma = 0.9
        discount_start = 2
        expected_return = 1.0 * (0.9**2) + 0.5 * (0.9**3)
        self.assertAlmostEqual(
            calc_return(rewards, gamma, discount_start), expected_return, places=6
        )

    def test_zero_rewards(self):
        """Test when all rewards are zero."""
        rewards = torch.tensor([0.0, 0.0, 0.0])
        gamma = 0.99
        expected_return = 0.0
        self.assertAlmostEqual(calc_return(rewards, gamma), expected_return, places=6)

    def test_high_gamma(self):
        """Test with a high discount factor close to 1."""
        rewards = torch.tensor([1.0, 1.0, 1.0])
        gamma = 0.999
        expected_return = 1.0 + 1.0 * 0.999 + 1.0 * (0.999**2)
        self.assertAlmostEqual(calc_return(rewards, gamma), expected_return, places=6)

    def test_dimensionality_check(self):
        """Test that a non-1D tensor raises an error."""
        rewards = torch.tensor([[1.0, 0.5], [0.25, 0.1]])  # 2D tensor
        gamma = 0.99
        with self.assertRaises(ValueError):
            calc_return(rewards, gamma)

    def test_empty_tensor(self):
        """Test with an empty tensor."""
        rewards = torch.tensor([])
        gamma = 0.99
        expected_return = 0.0
        self.assertAlmostEqual(calc_return(rewards, gamma), expected_return, places=6)

    def test_discount_offset(self):
        """Test with a non-zero discount offset."""
        rewards = torch.tensor([1.0, 0.5, 0.25])
        gamma = 0.9
        discount_start = 4
        expected_return = 1 * gamma**4 + 0.5 * gamma**5 + 0.25 * gamma**6
        self.assertAlmostEqual(
            calc_return(rewards, gamma, discount_start), expected_return, places=6
        )


if __name__ == "__main__":
    unittest.main()
