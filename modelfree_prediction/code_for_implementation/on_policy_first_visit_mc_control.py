"""On policy first visit Monte-carlo control."""


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
    :return: Value matrix of given policy in environment given
    """
    pass
