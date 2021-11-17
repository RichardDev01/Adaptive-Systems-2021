"""Agent class used for the maze."""


class Agent:
    """Agent class itself."""

    def __init__(self, policy):
        """Initialize agent with values."""
        self.policy = policy

    def interpret_world(self):
        """Process the world."""
        pass

    def value_function(self):
        """Calculate value of observation."""
        pass

    def __str__(self):
        return f"{self.policy=}\n" \
