"""Pure random policy class."""

from policy import Policy
from action import Action
import numpy as np


class PureRandomPolicy(Policy):
    """Pure random policy is a policy that takes action on a pure random base."""

    def __init__(self):
        """Init of random Policy."""
        self.visual = False
        self.visual_matrix = None

    def decide_action(self, observation):
        """
        Decide action based on pure random.

        :param observation: observation is a dict containing information about the environment. NOT USED FOR THIS POLICY
        :return: Random action chosen by policy
        """
        return np.random.choice([Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT])
