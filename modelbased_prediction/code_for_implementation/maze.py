"""Maze class file."""

import numpy as np


class Maze:
    """Class file for the maze as environment."""

    r_finish = 100          # Reward for finishing the maze
    r_move = -10            # Reward for moving a space
    r_backtracked = -5      # Reward for moving back over discovered path # TODO
    r_invalid_move = -20    # Reward for invalid move

    def __init__(self, agent, start_coord=(0, 0), end_coord=(3, 3)):
        """Create maze with initial values."""
        self.maze = np.array((4, 4))
        self.agent = agent
        self.start_coord = start_coord
        self.end_coord = end_coord

    def step(self):
        """Step function used for playing out decided actions."""
        pass

    def state(self):
        """State function for returning the world as a state to a policy."""
        pass

    def __str__(self):
        return f"{self.maze=}\n" \
               f"{self.agent=}\n" \
               f"{self.start_coord=}\n" \
               f"{self.end_coord=}"
