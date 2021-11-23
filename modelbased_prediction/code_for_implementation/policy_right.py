"""Pure random policy class."""

from policy import Policy
from action import Action


class RightPolicy(Policy):
    """Pure random policy is a policy that takes action on a pure random base."""

    def decide_action(self, observation):
        """Decide action based on pure random."""
        return Action.DOWN
