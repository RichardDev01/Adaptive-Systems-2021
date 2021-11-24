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

    def value_iteration(self, itera: int = 200, gamma: int = 1):
        """Calculate values for value function."""
        # Write environment to policy because of model based policy
        self.policy.agent = self
        self.policy.gamma = gamma
        self.policy.value_matrix = copy.copy(self.env.maze)

        # Iterate over values
        for i in range(itera):
            # Copy shape
            new_value_matrix = copy.deepcopy(self.policy.value_matrix)

            # Iterate over y and x
            for index_y, x in enumerate(self.policy.value_matrix):
                for index_x, _ in enumerate(x):
                    # Check if it is a terminal state
                    if (index_x, index_y) not in self.env.end_coord:
                        # Set the state that going to be used which is only agent position
                        state = (index_y, index_x)

                        # Let the value based policy decide the most greedy action
                        action = self.policy.decide_action({"agent_location": state})

                        # Reset environment to current state
                        self.env.reset(state)

                        # Use best available action and use Bellman equation
                        obs, r, _, _ = self.env.step(action)
                        new_value_matrix[state] = r + gamma * self.policy.value_matrix[obs["agent_location"]]
            # Check if there is a difference from previous iteration
            if np.allclose(self.policy.value_matrix, new_value_matrix):
                print(f"No difference from previous iteration")
                break
            # Update matrix
            self.policy.value_matrix = new_value_matrix

            # Visualise the update process # TODO

            print(self.policy.value_matrix)

    def get_action_from_policy(self, observation):
        """Get action from policy."""
        return self.policy.decide_action(observation)

    def __str__(self):
        return f"{self.policy=}\n" \
