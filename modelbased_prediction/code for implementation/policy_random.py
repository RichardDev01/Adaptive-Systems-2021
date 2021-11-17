from policy import Policy
from typing import List
from maze import Action
import numpy as np

class PureRandomPolicy(Policy):

    def decide_action(self, observation: List):
        return np.random.choice([Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT, Action.STAY]
