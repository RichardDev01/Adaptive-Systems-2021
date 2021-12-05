"""On policy first visit Monte-carlo control."""
import copy
import numpy as np


def on_policy_first_visit_mc_control(environment,
                                     iterations=10000,
                                     discount_rate=0.9,
                                     exploring_starts=False,
                                     epsilon=0.7):
    """
    On policy Monte Carlo control methods for updating given policy.

    Pseudo Code
    Input: a policy π to be evaluated
    Initialize:
        π ← an arbitrary ε-soft policy
        Q(s,a) ∈ R, arbitrarily, for all s ∈ S ∈ A(s)
        Returns(s, a) ← an empty list, for all s ∈ S ∈ A(s)

    Loop forever (for each episode):
        Generate an episode following π: S0,A0,R1, S1,A1,R2, . . . , ST−1,AT−1,RT
        G ← 0
        Loop for each step of episode, t = T −1, T −2, . . . , 0:
            G ← γG + Rt+1
            Unless St appears in S0, S1, . . . , St−1:
                Append G to Returns(St, At)
                Q(St, At) ← average(Returns(St, At))
                A* ← argmax a Q(St,a)                   (with ties broken arbitrarily)
                for all a ∈ A(St):
                                1 - ε + ε/|A(St)|   if a = A*
                    π(a|St) ←
                                ε/|A(St)|           if a ≠ A*

    :param environment: Environment of the simulation contains the agent with policy
    :param iterations: Loop amount for creating episodes
    :param discount_rate: Discount value used in algorithm
    :param exploring_starts: Enable or disable exploring starts
    :param epsilon: Parameter for E-soft policy
    :return:
    """

    #    Initialize:
    #    π ← an arbitrary ε-soft policy
    #    Q(s,a) ∈ R, arbitrarily, for all s ∈ S ∈ A(s)
    #    Returns(s, a) ← an empty list, for all s ∈ S ∈ A(s)

    environment.agent.policy.epsilon = epsilon

    dict_of_states = {}
    # Q table to policy
    q_table = [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]] for x in np.zeros_like(environment.maze)]
    environment.agent.policy.q_table = q_table

    array_estimates_policy = copy.copy(environment.maze)
    # Iterate over y and x
    for index_y, x in enumerate(array_estimates_policy):
        for index_x, _ in enumerate(x):
            # all_action = [Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT]
            state = (index_y, index_x)
            dict_of_states[state] = {"action_value": [0, 0, 0, 0],
                                     "rewards": [[0], [0], [0], [0]],
                                     "average": [0, 0, 0, 0]}

    # Loop forever (for each episode):
    #     Generate an episode following π: S0,A0,R1, S1,A1,R2, . . . , ST−1,AT−1,RT
    counter = 0
    episodes_dict = {}

    for i in range(iterations):
        episode_log = []
        environment.reset(random_start=exploring_starts)

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
        #         Append G to Returns(St, At)
        #         Q(St, At) ← average(Returns(St, At))
        #         A* ← argmax a Q(St,a)                   (with ties broken arbitrarily)
        #         for all a ∈ A(St):
        #                         1 - ε + ε/|A(St)|   if a = A*
        #             π(a|St) ←
        #                         ε/|A(St)|           if a ≠ A*
        big_g = 0
        inverted_episode_log = episode_log[::-1][1:]
        print(inverted_episode_log)
        # Step info[0] = State
        # Step info[1] = Action
        # Step info[2] = Reward
        for index, step_info in enumerate(inverted_episode_log):
            state_info = step_info[0]
            action_info = step_info[1]
            reward_info = step_info[2]

            big_g = discount_rate * big_g + reward_info
            if not state_info in [x[0] for x in inverted_episode_log[index + 1:]]:

                dict_of_states[step_info[0]]['rewards'][action_info].append(big_g)
                dict_of_states[step_info[0]]['average'][action_info] = np.average(dict_of_states[state_info]['rewards'][action_info])

        # value_matrix = copy.copy(environment.maze)
        # for key in dict_of_states.keys():
        #     value_matrix[key] = round(dict_of_states[key]['average'], 2)

    return environment.agent.policy.value_matrix
