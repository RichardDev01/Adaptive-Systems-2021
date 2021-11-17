"""Maze class file."""

import numpy as np


class Maze:
    """Class file for the maze as environment."""

    r_finish = 100          # Reward for finishing the maze
    r_move = -10            # Reward for moving a space
    r_backtracked = -5      # Reward for moving back over discovered path
    r_invalid_move = -20    # Reward for invalid move

    def __init__(self):
        """Create maze with initial values."""
        self.maze = np.array(4, 4)

    def step(self):
        """Step function used for playing out decided actions."""
        pass

    def state(self):
        """State function for returning the world as a state to a policy."""
        pass
