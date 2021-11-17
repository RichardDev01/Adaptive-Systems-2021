import numpy as np


class Action(int):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    STAY = 4


class Maze:

    r_finish = 100          # Reward for finishing the maze
    r_move = -10            # Reward for moving a space
    r_backtracked = -5      # Reward for moving back over discovered path
    r_invalid_move = -20    # Reward for invalid move

    def __init__(self):
        self.maze = np.array(4, 4)

    def step(self):
        pass

    def state(self):
        pass
