
def calculate_reward(state,next_state):
    return sum(state)-sum(next_state)
