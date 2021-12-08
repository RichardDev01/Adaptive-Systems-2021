"""Double Q Learning Learning."""
import copy
import numpy as np


def double_q_learning(environment, iterations=1000, discount_rate=0.9, alpha=0.1, exploring_starts=False, epsilon=0.7):
    """
    Q learning for policy control.

    Pseudo Code
    Algorithm parameter: step size α ∈ (0,1], small ε > 0
    Initialize Q1(s,a) and Q2(s,a), for all s ∈ S+,a ∈ A(s), arbitrarily except that V (terminal, *) = 0

    Loop for each step of episode
        Initialize S
        Loop for each step of episode:
            Choose A from S using policy ε-greedy in Q1 +Q2
            Take action A, observe R, S'
            With 0.5 probability:
                Q1(S,A) ← Q1(S,A) + α (R + γ Q2(S', maxa Q1(S',A)) - Q1(S,A))
            else:
                Q2(S,A) ← Q2(S,A) + α (R + γ Q1(S', maxa Q2(S',A)) - Q2(S,A))
            s ← S'
    until s is terminal

    :param environment: Environment of the simulation contains the agent with policy
    :param iterations: Loop amount for creating episodes
    :param discount_rate: Discount value used in algorithm
    :param alpha: alpha used in algorithm
    :param exploring_starts: Enable or disable exploring starts
    :param epsilon: Parameter for E-soft policy
    :return: Value matrix of given policy in environment given
    """
    q_table_1 = [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]] for x in np.zeros_like(environment.maze)]
    q_table_2 = [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]] for x in np.zeros_like(environment.maze)]
    environment.agent.policy.q_table_1 = q_table_1
    environment.agent.policy.q_table_2 = q_table_2

    for i in range(iterations):
        # Initialize S
        environment.reset(random_start=exploring_starts)
        # state = environment.get_state()

        # while not environment.done:
        #     # Choose A from S using policy derived from Q (e.g., ε-greedy)
        #     action = environment.agent.get_action_from_policy(state)
        #
        #     last_state = state
        #
        #     # Take action A, observe R, S'
        #     state_prime, reward, _, info = environment.step(action)
        #
        #     # Q(S,A) ← Q(S,A) + α (R + γ maxa Q(S',A) - Q(S,A))
        #     # max a Q(S',A)
        #     q_max_value = np.argmax(environment.agent.policy.q_table[state_prime['agent_location'][0]][state_prime['agent_location'][1]])
        #     environment.agent.policy.q_table[last_state['agent_location'][0]][last_state['agent_location'][1]][action] += alpha * (reward + discount_rate * environment.agent.policy.q_table[state_prime['agent_location'][0]][state_prime['agent_location'][1]][q_max_value] - environment.agent.policy.q_table[last_state['agent_location'][0]][last_state['agent_location'][1]][action])
        #
        #     # s ← S'
        #     state = state_prime

    print("done DQ learning")
    return environment.agent.policy.visualise_q_table()