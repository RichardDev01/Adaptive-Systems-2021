"""Q Learning Learning."""
import copy
import numpy as np


def q_learning(environment, iterations=1000, discount_rate=0.9, alpha=0.1, exploring_starts=False, epsilon=0.7):
    """
    Q learning for policy control.

    Pseudo Code
    Algorithm parameter: step size α ∈ (0,1], small ε > 0
    Initialize Q(s,a), for all s ∈ S+,a ∈ A(s), arbitrarily except that V (terminal, *) = 0

    Loop for each step of episode
        Initialize S
        Loop for each step of episode:
            Choose A from S using policy derived from Q (e.g., ε-greedy)
            Take action A, observe R, S'
            Q(S,A) ← Q(S,A) + α (R + γ maxa Q(S',A) - Q(S,A))
            s ← S'
    until s is terminal

    :param environment: Environment of the simulation contains the agent with policy
    :param iterations: Loop amount for creating episodes
    :param discount_rate: Discount value used in algorithm
    :param alpha:
    :param exploring_starts: Enable or disable exploring starts
    :param epsilon: Parameter for E-soft policy
    :return: Value matrix of given policy in environment given
    """
    q_table = [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]] for x in np.zeros_like(environment.maze)]
    environment.agent.policy.q_table = q_table

    # for i in range(iterations):
    #     environment.reset(random_start=exploring_starts)
    #     total_reward = 0
    #     observation = environment.get_state()
    #
    #     while not environment.done:
    #         # Decide an action according to the observation
    #         action = environment.agent.get_action_from_policy(observation)
    #         # Take action in the world
    #         last_observation = observation
    #         observation, reward, _, info = environment.step(action)
    #         # print(f"{observation['agent_location']=}\n{reward}")
    #         # Counting reward
    #         total_reward += reward
    #
    #         v_state = value_matrix[last_observation['agent_location']]
    #         v_state_prime = value_matrix[observation['agent_location']]
    #         value_matrix[last_observation['agent_location']] = round(v_state + alpha * (reward + discount_rate * v_state_prime - v_state), 2)
    #
    #     # print(f"{total_reward=}")
    return "q learning done"
