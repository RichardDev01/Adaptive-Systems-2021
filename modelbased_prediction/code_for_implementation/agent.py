"""Agent class used for the maze."""
import copy
import numpy as np


class Agent:
    """Agent class itself."""

    def __init__(self, policy, env=None):
        """Initialize agent with values."""
        self.policy = policy
        self.env = env

    def interpret_world(self):
        """Process the world."""
        pass

    def value_iteration(self, iter: int = 200, gamma: int = 1):
        """Calculate values for value function."""
        # coords_value = copy.copy(self.env.maze)
        self.policy.agent = self
        self.policy.value_matrix = copy.copy(self.env.maze)

        for i in range(iter):
            new_value_matrix = copy.deepcopy(self.policy.value_matrix)
            for index_y, x in enumerate(self.policy.value_matrix):
                for index_x, _ in enumerate(x):
                    if (index_x, index_y) not in self.env.end_coord:
                        state = (index_y, index_x)
                        action = self.policy.decide_action({"agent_location": state})
                        self.env.reset(state)
                        obs, r, _, _ = self.env.step(action)
                        new_value_matrix[state] = r + self.policy.value_matrix[obs["agent_location"]]
            self.policy.value_matrix = new_value_matrix
        print(self.policy.value_matrix)

    def get_action_from_policy(self, observation):
        """Get action from policy."""
        return self.policy.decide_action(observation)

    def __str__(self):
        return f"{self.policy=}\n" \
