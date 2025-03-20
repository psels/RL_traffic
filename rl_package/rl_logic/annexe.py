
def calculate_reward(state,next_state):
    n=len(state)//2
    return 3*(sum(state[:n])-sum(next_state[:n])) +sum(state[n:2*n])-sum(next_state[n:2*n])
