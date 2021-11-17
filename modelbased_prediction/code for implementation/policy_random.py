"""Pure random policy class."""

from policy import Policy
from typing import List
from maze import Action
import numpy as np


class PureRandomPolicy(Policy):
    """Pure random policy is a policy that takes action on a pure random base."""

    def decide_action(self, observation: List):
        """Decide action based on pure random."""
        return np.random.choice([Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT, Action.STAY])
