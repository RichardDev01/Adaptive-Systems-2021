"""This file is used for running the simulation."""

from maze import Maze
from policy_random import PureRandomPolicy
from agent import Agent

if __name__ == "__main__":
    policy = PureRandomPolicy
    a1 = Agent(policy)

    environment = Maze(a1)

    print(environment)
