"""This file is used for running the simulation for model free predictions."""

import cv2

import numpy as np
from maze import Maze
from policy_random import PureRandomPolicy
from policy_value_based import ValueBasedPolicy
from agent import Agent
import sys


if __name__ == "__main__":
    # Creating environment
    policy_pr = PureRandomPolicy()
    a1 = Agent(policy_pr)

    policy_vb = ValueBasedPolicy()
    a2 = Agent(policy_vb)

    environment_pr = Maze(a1, visualize=False)
    environment_vb = Maze(a2, visualize=True)

    # Value function for value based policy
    a2.value_iteration()

    # Creating variables that keep track of the simulation
    done = False
    total_reward = 0

    # Visualisation variables
    wait_key = 0
    window_name = 'Adaptive systems sim'

    # Get first observation for loop
    observation = environment_vb.get_state()

    # Create visualisation window
    if environment_vb.visualize:
        cv2.namedWindow(window_name, cv2.WINDOW_GUI_EXPANDED)
        cv2.resizeWindow(window_name, 800, 800)

    # Run simulation
    while not done:
        if environment_vb.visualize:
            # Render current time in simulation for visual output
            render = environment_vb.render()

            # Display render of current time in the environment
            cv2.imshow(window_name, cv2.cvtColor(np.array(render), cv2.COLOR_BGR2RGB))

            # Delay between renders of the simulation
            cv2.waitKey(wait_key)

            # On window [X] button press: stop the simulation and destroy the window
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                environment_vb.visualize = False
                break
        # Decide an action according to the observation
        action = environment_vb.agent.get_action_from_policy(observation)

        # Take action in the world
        observation, reward, done, info = environment_vb.step(action)

        # Counting reward
        total_reward += reward

    if environment_vb.visualize:
        # Render Last action of the simulation
        render = environment_vb.render()

        # Display render of current time in the environment
        cv2.imshow(window_name, cv2.cvtColor(np.array(render), cv2.COLOR_BGR2RGB))

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"{total_reward=}\ntime={environment_vb.sim_step}")
