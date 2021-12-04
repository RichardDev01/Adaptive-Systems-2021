"""Epsilon Soft greedy policy class."""

from policy import Policy
from action import Action
import numpy as np


class EpsilonSoftGreedyPolicy(Policy):
    """Epsilon Soft greedy policy."""

    def __init__(self, epsilon: 0.7):
        """
        Create Epsilon Soft greedy policy with parameters.

        :param epsilon: epsilon used in algorithm.
        """
        self.value_matrix = None
        self.epsilon = epsilon

    def decide_action(self, observation):
        """
        Decide action based on pure random.

        :param observation: observation is a dict containing information about the environment
        :return: Action chosen based on the observation
        """
        all_actions = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]

        if np.random.rand(1)[0] < self.epsilon:
            # Translate data class to coordinates
            translate_action_to_coord = {0: (-1, 0),
                                         1: (0, 1),
                                         2: (1, 0),
                                         3: (0, -1),
                                         4: (0, 0)}
            agent_pos = observation["agent_location"]
            value_states = []
            for index, action in enumerate(all_actions):
                next_agent_pos = agent_pos
                action_coord_delta_y, action_coord_delta_x = translate_action_to_coord[action]
                next_y = agent_pos[0] + action_coord_delta_y
                next_x = agent_pos[1] + action_coord_delta_x

                # Check if next action is possible in the maze
                if 0 <= next_y <= self.value_matrix.shape[1] - 1 and 0 <= next_x <= self.value_matrix.shape[0] - 1:
                    next_agent_pos = (next_y, next_x)

                value_states.append((action, self.value_matrix[next_agent_pos]))
            return max(value_states, key=lambda x: x[1])[0]
        else:
            return np.random.choice(all_actions)
