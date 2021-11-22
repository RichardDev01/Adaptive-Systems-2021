"""Maze class file."""

import numpy as np


class Maze:
    """Class file for the maze as environment."""

    def __init__(self, agent, start_coord=(0, 0), end_coord=(3, 3), visualize=False):
        """Create maze with initial values."""
        self.maze = np.zeros((4, 4))
        self.reward_map = np.array([[-1, -1, -1, 40],
                                    [-1, -1, -10, -10],
                                    [-1, -1, -1, -1],
                                    [10, -2, -1, -1]])
        self.visited_places_map = np.zeros((4, 4))
        self.agent = agent
        self.start_coord = start_coord
        self.end_coord = end_coord
        self.visualize = visualize

    def step(self):
        """Step function used for playing out decided actions."""
        pass

    def state(self):
        """State function for returning the world as a state to a policy."""
        pass

    def render(self):
        pass

    def __str__(self):
        """Return for debugging."""
        return f"{self.maze=}\n" \
               f"{self.reward_map=}\n" \
               f"{self.agent=}\n" \
               f"{self.start_coord=}\n" \
               f"{self.end_coord=}"
