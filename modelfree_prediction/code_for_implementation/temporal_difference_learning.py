"""Temporal Difference Learning."""
import copy


def tem_dif_ler(environment, iterations=1000, discount_rate=0.9, alpha=0.1, exploring_starts=False):
    """
    Estimate value policy using Tabular Temporal Difference Learning.

    Pseudo Code
    Input: a policy π to be evaluated
    Algorithm parameter: step size α ∈ (0,1]
    Initialize V(s), for all s ∈ S+, arbitrarily except that V (terminal) = 0

    Loop for each step of episode
        Initialize S
        Loop for each step of episode:
        A ← action given by π for S
        Take action A, observe R, S'
        V(S) ← V(S) + α * (R + γV(s') - V(S))
        s ← S'
    until s is terminal

    :param environment: Environment of the simulation contains the agent with policy
    :param iterations: Loop amount for creating episodes
    :param discount_rate: Discount value used in algorithm
    :param alpha: alpha used in algorithm
    :param exploring_starts: Enable or disable exploring starts
    :return: Value matrix of given policy in environment given
    """
    # Initialize V(s), for all s ∈ S+, arbitrarily except that V (terminal) = 0
    value_matrix = copy.copy(environment.maze)

    # Loop for each step of episode
    for i in range(iterations):
        environment.reset(random_start=exploring_starts)
        # Initialize S
        observation = environment.get_state()

        while not environment.done:
            # A ← action given by π for S
            action = environment.agent.get_action_from_policy(observation)

            last_observation = observation

            # Take action A, observe R, S'
            observation, reward, _, _ = environment.step(action)

            v_state = value_matrix[last_observation['agent_location']]
            v_state_prime = value_matrix[observation['agent_location']]

            # V(S) ← V(S) + α * (R + γV(s') - V(S))
            value_matrix[last_observation['agent_location']] = round(v_state + alpha * (reward + discount_rate * v_state_prime - v_state), 2)

        # print(f"{total_reward=}")
    return value_matrix
