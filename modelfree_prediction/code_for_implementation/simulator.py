"""This file is used for running the simulation for model free predictions."""
import sys
from maze import Maze
from policy_random import PureRandomPolicy
from policy_value_based import ValueBasedPolicy
from policy_epsilon_soft_greedy import EpsilonSoftGreedyPolicy
from agent import Agent

from first_visit_mc_evaluation import first_visit_mc
from temporal_difference_learning import tem_dif_ler
from on_policy_first_visit_mc_control import on_policy_first_visit_mc_control

import os

if __name__ == "__main__":
    if len(sys.argv) < 2:
        path = os.getcwd()
        os.chdir('modelfree_prediction\code_for_implementation')

    # Creating environment
    policy_pr = PureRandomPolicy()
    a1 = Agent(policy_pr)

    policy_vb = ValueBasedPolicy()
    a2 = Agent(policy_vb)

    # Creating epsilon soft greedy policy
    policy_epsilon_greedy = EpsilonSoftGreedyPolicy()
    a3 = Agent(policy_epsilon_greedy)

    environment_pr = Maze(a1, visualize=False)
    environment_vb = Maze(a2, visualize=False)
    environment_eg = Maze(a3, visualize=False)

    # Value function for value based policy
    a2.value_iteration()
    # a2.save_value_matrix('policy_saves/value_iteration_matrix.csv')
    # print(a2.policy.value_matrix)

    # Load optimal value matrix
    a2.load_value_matrix('policy_saves/value_iteration_matrix.csv')
    # print(a2.policy.value_matrix)



    # Creating variables that keep track of the simulation
    done = False
    total_reward = 0
    try:
        if int(sys.argv[1]) == 0:
            iterations = 10000
            discount_rate = 1
            exploring_starts = True
            print(f"Value based poly ;{iterations=}\t{discount_rate=}\t{exploring_starts=}\nOutcome\n")

            print(first_visit_mc(environment_vb,
                                 iterations=iterations,
                                 discount_rate=discount_rate,
                                 exploring_starts=exploring_starts), "\n")

            discount_rate = 0.9

            print(f"Value based poly ;{iterations=}\t{discount_rate=}\t{exploring_starts=}\nOutcome\n")

            print(first_visit_mc(environment_vb,
                                 iterations=iterations,
                                 discount_rate=discount_rate,
                                 exploring_starts=exploring_starts), "\n")

            discount_rate = 1

            print(f"Random based poly ;{iterations=}\t{discount_rate=}\t{exploring_starts=}\nOutcome\n")

            print(first_visit_mc(environment_pr,
                                 iterations=iterations,
                                 discount_rate=discount_rate,
                                 exploring_starts=exploring_starts), "\n")

            discount_rate = 0.9

            print(f"Random based poly ;{iterations=}\t{discount_rate=}\t{exploring_starts=}\nOutcome\n")

            print(first_visit_mc(environment_pr,
                                 iterations=iterations,
                                 discount_rate=discount_rate,
                                 exploring_starts=exploring_starts), "\n")

        if int(sys.argv[1]) == 1:
            iterations = 10000
            discount_rate = 1
            alpha = 0.1
            exploring_starts = True
            print(
                f"Value based poly Temporal Difference Learning\n{iterations=}\t{discount_rate=}\t{alpha=}\t{exploring_starts=}\nOutcome\n")
            print(tem_dif_ler(environment_vb,
                              iterations=iterations,
                              discount_rate=discount_rate,
                              alpha=alpha,
                              exploring_starts=exploring_starts))
            discount_rate = 0.9
            print(
                f"Value based poly Temporal Difference Learning\n{iterations=}\t{discount_rate=}\t{alpha=}\t{exploring_starts=}\nOutcome\n")
            print(tem_dif_ler(environment_vb,
                              iterations=iterations,
                              discount_rate=discount_rate,
                              alpha=alpha,
                              exploring_starts=exploring_starts))

            discount_rate = 1
            alpha = 0.1
            exploring_starts = True
            print(
                f"Random based poly Temporal Difference Learning\n{iterations=}\t{discount_rate=}\t{alpha=}\t{exploring_starts=}\nOutcome\n")
            print(tem_dif_ler(environment_pr,
                              iterations=iterations,
                              discount_rate=discount_rate,
                              alpha=alpha,
                              exploring_starts=exploring_starts))

            discount_rate = 0.9
            print(
                f"Random based poly Temporal Difference Learning\n{iterations=}\t{discount_rate=}\t{alpha=}\t{exploring_starts=}\nOutcome\n")
            print(tem_dif_ler(environment_pr,
                              iterations=iterations,
                              discount_rate=discount_rate,
                              alpha=alpha,
                              exploring_starts=exploring_starts))

        if int(sys.argv[1]) == 2:
            # Load optimal value matrix
            a3.load_value_matrix('policy_saves/best_paths_custom.csv')
            # print(a3.policy.value_matrix)

            iterations = 1
            discount_rate = 1
            alpha = 0.1
            exploring_starts = False
            # epsilon = 0.7
            epsilon = 1
            print(
                f"on policy control e soft greedy policy\n{iterations=}\t{discount_rate=}\t{alpha=}\t{exploring_starts=}\t{epsilon=}\nOutcome\n")
            print(on_policy_first_visit_mc_control(environment_eg,
                                                   iterations=iterations,
                                                   discount_rate=discount_rate,
                                                   exploring_starts=exploring_starts,
                                                   epsilon=epsilon))

            # print("3")
    except IndexError:
        print("")
