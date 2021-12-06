"""Epsilon Soft greedy policy class."""

from policy import Policy
from action import Action
import numpy as np
import cv2

from PIL import Image, ImageDraw

from pathlib import Path
textures_path = Path(__file__) / '..' / 'visualisation' / 'textures'


class EpsilonSoftGreedyPolicy(Policy):
    """Epsilon Soft greedy policy."""

    def __init__(self, epsilon=0.7):
        """
        Create Epsilon Soft greedy policy with parameters.

        :param epsilon: epsilon used in algorithm.
        """
        self.value_matrix = None
        self.epsilon = epsilon
        self.q_table = None

    def decide_action(self, observation):
        """
        Decide action based on pure random.

        :param observation: observation is a dict containing information about the environment
        :return: Action chosen based on the observation
        """
        all_actions = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]

        if np.random.rand(1)[0] < self.epsilon:
            agent_pos = observation["agent_location"]
            max_value = max(self.q_table[agent_pos[0]][agent_pos[1]])
            index_action = self.q_table[agent_pos[0]][agent_pos[1]].index(max_value)
            chosen_action = index_action
            return chosen_action
        else:
            # print("random")
            return np.random.choice(all_actions)

    def visualise_q_table(self):
        """Visualise q table to img."""
        # print(self.q_table)
        # action_to_string_dict = {0: 'up',
        #                          1: 'right',
        #                          2: 'down',
        #                          3: 'left', }
        #
        # new_matrix = np.zeros_like(self.value_matrix)
        # for index_y, y in enumerate(self.q_table):
        #     for index_x, x in enumerate(y):
        #         # print(np.argmax(x))
        #         new_matrix[index_y, index_x] = np.argmax(x)
        #
        # # print(new_matrix)
        #
        # translated_matrix = np.zeros_like(new_matrix, dtype=str)
        # for index_y, y in enumerate(self.q_table):
        #     for index_x, x in enumerate(y):
        #         # print(np.argmax(x))
        #         translated_matrix[index_y, index_x] = action_to_string_dict[np.argmax(x)]

        triangles = Image.open(textures_path / "baground_tiles_for_q.png")

        tile_size_width, tile_size_height = triangles.size

        maze = np.ndarray((4, 4))

        width = maze.shape[0] * tile_size_width
        height = maze.shape[1] * tile_size_height
        background = Image.new(mode="RGB", size=(width, height))

        for height_row, width_values in enumerate(maze):
            for index, value in enumerate(width_values):
                background.paste(triangles, (index * tile_size_width, height_row * tile_size_height), triangles)

        # copy_background = background.copy()

        off_set_text = [(-40, 0), (0, 40), (40, 0), (0, -40)]

        for height_row, width_values in enumerate(self.q_table):
            for index, q_values in enumerate(width_values):
                for index_v, value in enumerate(q_values):
                    #             print(value)
                    ImageDraw.Draw(background).text(
                        (height_row * tile_size_height + tile_size_height / 2.5 + off_set_text[index_v][1],
                         index * tile_size_width + tile_size_width / 2.5 + off_set_text[index_v][0]),
                        f"{round(value, 2)}", fill=(255, 0, 0, 255))
                direction = np.argmax(q_values)
                ImageDraw.Draw(background).text(
                    (height_row * tile_size_height + tile_size_height / 2 + off_set_text[direction][1],
                     index * tile_size_width + tile_size_width / 2.5 + off_set_text[direction][0]),
                    "\n|=|", fill=(255, 0, 0, 255))

        cv2.imshow('Q-table visualised', cv2.cvtColor(np.array(background), cv2.COLOR_BGR2RGB))

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return self.q_table
