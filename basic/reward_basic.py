def calculate_reward(state, next_state):
    return -(sum(next_state) - sum(state))