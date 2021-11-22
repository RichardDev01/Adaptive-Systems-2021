"""Maze class file."""

import numpy as np
from visualisation.enviroment_render import render_background, render_in_step


class Maze:
    """Class file for the maze as environment."""

    def __init__(self, agent, start_coord=(0, 0), end_coord=(3, 3), visualize=False, done=False):
        """Create maze with initial values."""
        self.maze = np.zeros((4, 4))
        self.reward_map = np.array([[-1, -1, -1, 40],
                                    [-1, -1, -10, -10],
                                    [-1, -1, -1, -1],
                                    [10, -2, -1, -1]])
        self.occupied_map = np.zeros((4, 4))
        self.agent = agent
        self.start_coord = start_coord
        self.end_coord = end_coord

        self.occupied_map[start_coord] = 1  # Agent occupation

        self.done = done

        self.sim_step = 0

        self.last_action_agent = None

        self.visualize = visualize
        if self.visualize:
            self.rendered_background = render_background(self)
        else:
            self.rendered_background = None

    def step(self, action):
        """Step function used for playing out decided actions."""
        observation = self.get_state()
        reward = -1
        self.sim_step += 1

        self.last_action_agent = action

        # self.done = True # debug
        return observation, reward, self.done, {}

    def get_state(self):
        """State function for returning the world as a state to a policy."""
        return {"occupied_map": self.occupied_map,
                "maze_map": self.maze}

    def render(self):
        return render_in_step(self)
        # return self.rendered_background
        # pass

    def __str__(self):
        """Return for debugging."""
        return f"{self.maze=}\n" \
               f"{self.reward_map=}\n" \
               f"{self.agent=}\n" \
               f"{self.start_coord=}\n" \
               f"{self.end_coord=}"
