"""This file is used for running the simulation."""

import cv2

import numpy as np
from PIL import Image
from maze import Maze
from policy_random import PureRandomPolicy
from agent import Agent

if __name__ == "__main__":
    policy = PureRandomPolicy
    a1 = Agent(policy)

    environment = Maze(a1, visualize=True)

    # print(environment)
    done = False
    total_reward = 0

    wait_key = 250
    window_name = 'Adaptive systems sim'

    observation = environment.get_state()
    print(observation)

    while not done:
        # For every agent, decide an action according to the observation
        action = a1.get_action_from_policy(observation)

        print(f"{action=}")

        observation, reward, done, info = environment.step(action)

        total_reward += reward

        if environment.visualize:
            # Render current time in simulation for visual output
            render = environment.render()

            # Display render of current time in the environment
            cv2.imshow(window_name, cv2.cvtColor(np.array(render), cv2.COLOR_BGR2RGB))

            # Delay between renders of the simulation
            cv2.waitKey(wait_key)

            # On window [X] button press: stop the simulation and destroy the window
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

    if environment.visualize:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
