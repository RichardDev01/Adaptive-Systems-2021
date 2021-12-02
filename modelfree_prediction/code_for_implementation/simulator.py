"""This file is used for running the simulation for model free predictions."""

import cv2

import numpy as np
from maze import Maze
from policy_random import PureRandomPolicy
from policy_value_based import ValueBasedPolicy
from agent import Agent
import sys
import copy


def first_visit_mc(environment, iterations=10000, discount_rate=0.9, random_spawn=False):
    """
    Monte Carlo methods for learning the state-value function for a given policy.

    Pseudo Code
    Input: a policy π to be evaluated
    Initialize:
        V (s) ∈ R, arbitrarily, for all s ∈ S
        Returns(s) ← an empty list, for all s ∈ S

    Loop forever (for each episode):
        Generate an episode following π: S0,A0,R1, S1,A1,R2, . . . , ST−1,AT−1,RT
        G ← 0
        Loop for each step of episode, t = T −1, T −2, . . . , 0:
            G ← γG + Rt+1
            Unless St appears in S0, S1, . . . , St−1:
                Append G to Returns(St)
                V (St) ← average(Returns(St))


    :return:
    """
    #    Initialize:
    #    V (s) ∈ R, arbitrarily, for all s ∈ S
    #    Returns(s) ← an empty list, for all s ∈ S

    dict_of_states = {}
    array_estimates_policy = copy.copy(environment.maze)
    # Iterate over y and x
    for index_y, x in enumerate(array_estimates_policy):
        for index_x, _ in enumerate(x):
            # all_action = [Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT]
            state = (index_y, index_x)
            dict_of_states[state] = {"action_value": [0, 0, 0, 0], "rewards": [0], "average": 0}

    # Loop forever (for each episode):
    #     Generate an episode following π: S0,A0,R1, S1,A1,R2, . . . , ST−1,AT−1,RT
    counter = 0
    episodes_dict = {}

    for i in range(iterations):
        episode_log = []
        environment.reset(random_start=random_spawn)

        total_reward = 0
        # Get first observation for loop
        first_observation = environment.get_state()
        action = environment.agent.get_action_from_policy(first_observation)
        observation, reward, _, info = environment.step(action)
        total_reward += reward

        episode_log.append([first_observation['agent_location'], action, reward])

        while not environment.done:
            counter += 1
            # Decide an action according to the observation
            action = environment.agent.get_action_from_policy(observation)
            last_observation = observation
            # Take action in the world
            observation, reward, _, info = environment.step(action)
            episode_log.append([last_observation['agent_location'], action, reward])

            # Counting reward
            total_reward += reward
        episodes_dict[i] = episode_log
        episodes_dict[i].append([total_reward, environment.sim_step])

        # G ← 0
        # Loop for each step of episode, t = T −1, T −2, . . . , 0:
        #     G ← γG + Rt+1
        #     Unless St appears in S0, S1, . . . , St−1:
        #         Append G to Returns(St)
        #         V (St) ← average(Returns(St))

        return_list = []
        big_g = 0
        inverted_episode_log = episode_log[::-1][1:]
        for index, step_info in enumerate(inverted_episode_log):
            big_g = discount_rate * big_g + step_info[2]
            if not step_info[0] in [x[0] for x in inverted_episode_log[index + 1:]]:
                return_list.append((step_info[0], big_g))
                dict_of_states[step_info[0]]['rewards'].append(big_g)
                dict_of_states[step_info[0]]['average'] = np.average(dict_of_states[step_info[0]]['rewards'])

        value_matrix = copy.copy(environment.maze)
        for key in dict_of_states.keys():
            value_matrix[key] = round(dict_of_states[key]['average'], 2)

    # print(value_matrix)

    # print(episodes_dict)
    return value_matrix


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

    print("\n----\n")

    # print(environment_vb.agent.first_visit_mc())
    # print(environment_vb.first_visit_mc())

    print(first_visit_mc(environment_vb, iterations=30000, discount_rate=0.9, random_spawn=True))

    # # Visualisation variables
    # wait_key = 0
    # window_name = 'Adaptive systems sim'
    #
    # # Get first observation for loop
    # observation = environment_vb.get_state()
    #
    # # Run simulation
    # while not done:
    #     # Decide an action according to the observation
    #     action = environment_vb.agent.get_action_from_policy(observation)
    #
    #     # Take action in the world
    #     observation, reward, done, info = environment_vb.step(action)
    #
    #     # Counting reward
    #     total_reward += reward
    #
    # print(f"{total_reward=}\ntime={environment_vb.sim_step}")

    # # Create visualisation window
    # if environment_vb.visualize:
    #     cv2.namedWindow(window_name, cv2.WINDOW_GUI_EXPANDED)
    #     cv2.resizeWindow(window_name, 800, 800)
    #
    # # Run simulation
    # while not done:
    #     if environment_vb.visualize:
    #         # Render current time in simulation for visual output
    #         render = environment_vb.render()
    #
    #         # Display render of current time in the environment
    #         cv2.imshow(window_name, cv2.cvtColor(np.array(render), cv2.COLOR_BGR2RGB))
    #
    #         # Delay between renders of the simulation
    #         cv2.waitKey(wait_key)
    #
    #         # On window [X] button press: stop the simulation and destroy the window
    #         if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
    #             environment_vb.visualize = False
    #             break
    #     # Decide an action according to the observation
    #     action = environment_vb.agent.get_action_from_policy(observation)
    #
    #     # Take action in the world
    #     observation, reward, done, info = environment_vb.step(action)
    #
    #     # Counting reward
    #     total_reward += reward
    #
    # if environment_vb.visualize:
    #     # Render Last action of the simulation
    #     render = environment_vb.render()
    #
    #     # Display render of current time in the environment
    #     cv2.imshow(window_name, cv2.cvtColor(np.array(render), cv2.COLOR_BGR2RGB))
    #
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #
    # print(f"{total_reward=}\ntime={environment_vb.sim_step}")
