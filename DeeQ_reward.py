import numpy as np

reward = 0
total_reward = 0
list_values = [[0,0,0,0]]
penal_mult1 = 0
penal_mult2 = 0
penal_mult3 = 0
penal_mult4 = 0

def calculate_reward(values, reward, total_reward):

    prev_val = np.sum(np.array(list_values[-1]))

    list_values.append(values)

    actual_val = np.sum(np.array(list_values[-1]))

    total_reward += np.sum(actual_val)

    reward = prev_val - actual_val

    return  reward, total_reward, prev_val