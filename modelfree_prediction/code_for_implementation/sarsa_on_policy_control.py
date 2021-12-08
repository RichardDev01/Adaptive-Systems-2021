"""Sarsa on policy Temporal Difference control."""
import copy
import numpy as np


def sarsa_tem_dif_ler(environment, iterations=1000, discount_rate=0.9, alpha=0.1, exploring_starts=False, epsilon=0.7):
    """
    Policy control using SARSA temporal difference.

    Pseudo Code
    Algorithm parameter: step size α ∈ (0,1], ε > 0
    Initialize Q(s,a), for all s ∈ S+,a ∈ A(s), arbitrarily except that V (terminal, *) = 0

    Loop for each step of episode
        Initialize S
        Choose A from S using policy derived from Q (e.g., ε-greedy)
            Loop for each step of episode:
            Take action A, observe R, S'
            Choose A' from S' using policy derived from Q (e.g., ε-greedy)
            Q(S,A) ← Q(S,A) + α (R + γQ(S',A') - Q(S,A))
            S ← S'; A ← A'
        until s is terminal

    :param environment: Environment of the simulation contains the agent with policy
    :param iterations: Loop amount for creating episodes
    :param discount_rate: Discount value used in algorithm
    :param alpha:
    :param exploring_starts: Enable or disable exploring starts
    :param epsilon: Parameter for E-soft policy
    :return: Value matrix of given policy in environment given
    """
    value_matrix = copy.copy(environment.maze)

    q_table = [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]] for x in np.zeros_like(environment.maze)]
    environment.agent.policy.q_table = q_table

    translate_action_to_coord = {0: (-1, 0),
                                 1: (0, 1),
                                 2: (1, 0),
                                 3: (0, -1),
                                 4: (0, 0)}

    for i in range(iterations):
        environment.reset(random_start=exploring_starts)
        total_reward = 0

        # Initialize S
        state = environment.get_state()
        # Choose A from S using policy derived from Q (e.g., ε-greedy)
        action = environment.agent.get_action_from_policy(state)

        while not environment.done:

            last_state = state
            # Take action A, observe R, S'
            state_prime, reward, _, _ = environment.step(action)

            # Choose A' from S' using policy derived from Q (e.g., ε-greedy)
            # state_prime = observation
            action_prime = environment.agent.get_action_from_policy(state_prime)

            # Get Q_value for prime state
            action_coord_delta_y, action_coord_delta_x = translate_action_to_coord[action]
            next_y = state_prime['agent_location'][0] + action_coord_delta_y
            next_x = state_prime['agent_location'][1] + action_coord_delta_x
            print(f"{state=}\t{action_prime=}\t")

            # Check if next action is possible in the maze
            if 0 <= next_y <= environment.maze.shape[1] - 1 and 0 <= next_x <= environment.maze.shape[0] - 1:
                state_prime['agent_location'] = (next_y, next_x)

            # q_value_from_prime_state = environment.agent.policy.q_table[(next_y, next_x)]
            q_value_from_prime_state = environment.agent.policy.q_table[state_prime['agent_location'][0]][state_prime['agent_location'][1]]
            # print(f"{q_value_from_prime_state=}")

            q_value_from_state = environment.agent.policy.q_table[last_state['agent_location'][0]][last_state['agent_location'][1]]
            # print(f"{q_value_from_state=}")

            # Q(S,A) ← Q(S,A) + α (R + γQ(S',A') - Q(S,A))
            environment.agent.policy.q_table[last_state['agent_location'][0]][last_state['agent_location'][1]] = q_value_from_state + alpha * (reward + discount_rate * q_value_from_prime_state - q_value_from_state)

            total_reward += reward

            # S ← S'; A ← A'
            state = state_prime
            action = action_prime
            # print(f"{state=}")
        # print(f"{total_reward=}")
        #     print("loop")
        print(environment.agent.policy.q_table)
    return environment.agent.policy.visualise_q_table()
