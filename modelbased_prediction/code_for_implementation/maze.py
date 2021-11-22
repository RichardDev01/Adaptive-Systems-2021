"""Maze class file."""

import numpy as np
from visualisation.enviroment_render import render_background, render_in_step


class Maze:
    """Class file for the maze as environment."""

    def __init__(self, agent, start_coord=(0, 0), end_coords=[(0, 3), (3, 0)], visualize=False, done=False):
        """Create maze with initial values."""
        self.maze = np.zeros((4, 4))
        self.reward_map = np.array([[-1, -1, -1, 40],
                                    [-1, -1, -10, -10],
                                    [-1, -1, -1, -1],
                                    [10, -2, -1, -1]])
        self.occupied_map = np.zeros((4, 4))
        self.agent = agent
        self.start_coord = start_coord
        self.end_coord = end_coords

        # self.occupied_map[start_coord] = 1  # Agent occupation

        self.agent_location = start_coord

        self.done = done

        self.sim_step = 0

        self.last_action_agent = None

        self.visualize = visualize
        if self.visualize:
            self.rendered_background = render_background(self)
        else:
            self.rendered_background = None
        self.total_reward = 0

    def step(self, action):
        """Step function used for playing out decided actions."""
        observation = self.get_state()

        self.sim_step += 1
        self.last_action_agent = action

        translate_action_to_coord = {0: (-1, 0),
                                     1: (0, 1),
                                     2: (1, 0),
                                     3: (0, -1),
                                     4: (0, 0)}

        action_coord_delta_y, action_coord_delta_x = translate_action_to_coord[action]

        next_y = self.agent_location[0] + action_coord_delta_y
        next_x = self.agent_location[1] + action_coord_delta_x

        if 0 <= next_y <= self.maze.shape[1] - 1 and 0 <= next_x <= self.maze.shape[0] - 1:
            self.agent_location = (next_y, next_x)

        reward = self.reward_map[self.agent_location]
        self.total_reward += reward

        if self.agent_location in self.end_coord:
            self.done = True

        return observation, reward, self.done, {}

    def get_state(self):
        """State function for returning the world as a state to a policy."""
        return {"occupied_map": self.occupied_map,
                "maze_map": self.maze}

    def render(self):
        """Render function for visualizing the maze."""
        return render_in_step(self)

    def __str__(self):
        """Return for debugging."""
        return f"{self.maze=}\n" \
               f"{self.reward_map=}\n" \
               f"{self.agent=}\n" \
               f"{self.start_coord=}\n" \
               f"{self.end_coord=}"
