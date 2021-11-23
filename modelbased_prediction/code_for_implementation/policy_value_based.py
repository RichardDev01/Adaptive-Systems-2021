"""Value based policy class."""

from policy import Policy
from action import Action

class ValueBasedPolicy(Policy):
    """Pure random policy is a policy that takes action on a pure random base."""

    def __init__(self, gamma = 1):
        # self.value_matrix = np.zeros(maze_shape, dtype=float)
        self.value_matrix = None
        self.agent = None
        self.gamma = gamma

    def decide_action(self, observation):
        """Decide action based on pure random."""
        all_action = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
        outcome = []
        for action in all_action:
            self.agent.env.reset(observation["agent_location"])
            obs, r, _, _ = self.agent.env.step(action)
            outcome.append((action, r, obs))
        return max(outcome, key=lambda x: x[1] + self.gamma * self.value_matrix[x[2]["agent_location"]])[0]
