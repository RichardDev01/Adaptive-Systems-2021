"""Epsilon Soft greedy policy class."""

from policy import Policy
from action import Action
import numpy as np


class EpsilonSoftGreedyPolicy(Policy):
    """Epsilon Soft greedy policy."""

    def __init__(self, epsilon=0.7):
        """
        Create Epsilon Soft greedy policy with parameters.

        :param epsilon: epsilon used in algorithm.
        """
        self.value_matrix = None
        self.epsilon = epsilon
        self.q_table = None

    def decide_action(self, observation):
        """
        Decide action based on pure random.

        :param observation: observation is a dict containing information about the environment
        :return: Action chosen based on the observation
        """
        all_actions = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]

        if np.random.rand(1)[0] < self.epsilon:
            agent_pos = observation["agent_location"]
            max_value = max(self.q_table[agent_pos[0]][agent_pos[1]])
            index_action = self.q_table[agent_pos[0]][agent_pos[1]].index(max_value)
            chosen_action = index_action
            return chosen_action
        else:
            # print("random")
            return np.random.choice(all_actions)
