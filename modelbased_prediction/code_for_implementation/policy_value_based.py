"""Value based policy class."""

from policy import Policy
from action import Action


class ValueBasedPolicy(Policy):
    """Pure random policy is a policy that takes action on a pure random base."""

    def __init__(self, gamma=1, visuals=True):
        """
        Create Value based policy with parameters.

        :param gamma: Gamma is the discount value is this context
        :param visuals: This parameter is used to check if we want to visualise
        """
        self.value_matrix = None
        self.visual_matrix = None
        self.agent = None
        self.gamma = gamma
        self.visual = visuals

    def decide_action(self, observation):
        """
        Decide action based on pure random.

        :param observation: observation is a dict containing information about the environment
        :return: Action chosen based on the observation
        """
        all_action = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
        outcome = []
        # Get all values from every action possible
        for action in all_action:
            self.agent.env.reset(observation["agent_location"])
            obs, r, _, _ = self.agent.env.step(action)
            outcome.append((action, r, obs))

        # Return best value using the Bellman equation
        return max(outcome, key=lambda x: x[1] + self.gamma * self.value_matrix[x[2]["agent_location"]])[0]
