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

    iterations = 10000
    discount_rate = 1
    exploring_starts = True
    print(f"Value based poly ;{iterations=}\t{discount_rate=}\t{exploring_starts=}\nOutcome\n")

    print(first_visit_mc(environment_vb, iterations=iterations, discount_rate=discount_rate, exploring_starts=exploring_starts), "\n")

    discount_rate = 0.9

    print(f"Value based poly ;{iterations=}\t{discount_rate=}\t{exploring_starts=}\nOutcome\n")

    print(first_visit_mc(environment_vb, iterations=iterations, discount_rate=discount_rate, exploring_starts=exploring_starts), "\n")

    discount_rate = 1

    print(f"Random based poly ;{iterations=}\t{discount_rate=}\t{exploring_starts=}\nOutcome\n")

    print(first_visit_mc(environment_pr, iterations=iterations, discount_rate=discount_rate, exploring_starts=exploring_starts), "\n")

    discount_rate = 0.9

    print(f"Random based poly ;{iterations=}\t{discount_rate=}\t{exploring_starts=}\nOutcome\n")

    print(first_visit_mc(environment_pr, iterations=iterations, discount_rate=discount_rate, exploring_starts=exploring_starts), "\n")
