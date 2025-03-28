def calculate_reward(state, next_state):
    """
    Computes the reward based on traffic state change.

    The state vector is assumed to be structured as:
    - First half: number of halted vehicles per lane.
    - Second half: total number of vehicles per lane.

    Reward =
        3 x (reduction in halted vehicles)
      + 1 x (reduction in total vehicles)

    This formulation encourages:
    - Traffic fluidity by prioritizing the clearing of halted vehicles (x3).
    - Global throughput by reducing total vehicle count.
    - Fairness by helping vehicles that have been waiting too long to move.

    Args:
        state (List[float]): state before the step
        next_state (List[float]): state after the step

    Returns:
        float: computed reward
    """
    n = len(state) // 2
    return 3 * (sum(state[:n]) - sum(next_state[:n])) + (sum(state[n:2*n]) - sum(next_state[n:2*n]))
