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



def calculate_reward_V2(values, reward, total_reward, penal_mult1, penal_mult2, penal_mult3, penal_mult4):

    print(values)

    # if values[0] != 0:
    #     penal_mult1 += 0.1
    # if values[1] != 0:
    #     penal_mult2 += 0.1
    # if values[2] != 0:
    #     penal_mult3 += 0.1
    # if values[3] != 0:
    #     penal_mult4 += 0.1

    # values_penal = []

    # values_penal.append(values[0] + values[0] * penal_mult1)
    # values_penal.append(values[1] + values[1] * penal_mult2)
    # values_penal.append(values[2] + values[2] * penal_mult3)
    # values_penal.append(values[3] + values[3] * penal_mult4)

    # prev_val = np.sum(np.array(list_values[-1]))

    # list_values.append(values_penal)

    # actual_val = np.sum(np.array(list_values[-1]))

    # total_reward += np.sum(actual_val)

    # reward = prev_val - actual_val

    # return  reward, total_reward, prev_val


calculate_reward_V2(values, reward, total_reward, penal_mult1, penal_mult2, penal_mult3, penal_mult4)
