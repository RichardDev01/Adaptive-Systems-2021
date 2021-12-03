"""This file is used for running the simulation for model free predictions."""

from maze import Maze
from policy_random import PureRandomPolicy
from policy_value_based import ValueBasedPolicy
from agent import Agent

from first_visit_mc_evaluation import first_visit_mc


if __name__ == "__main__":
    # Creating environment
    policy_pr = PureRandomPolicy()
    a1 = Agent(policy_pr)

    policy_vb = ValueBasedPolicy()
    a2 = Agent(policy_vb)

    environment_pr = Maze(a1, visualize=False)
    environment_vb = Maze(a2, visualize=False)

    # Value function for value based policy
    a2.value_iteration()

    # Creating variables that keep track of the simulation
    done = False
    total_reward = 0

    print("")

    print(first_visit_mc(environment_vb, iterations=30000, discount_rate=0.9, exploring_starts=True))
