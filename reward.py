import numpy as np


penal_mult1 = 0
penal_mult2 = 0
penal_mult3 = 0
penal_mult4 = 0

list_values = [[0,0,0,0]]




def calculate_reward(state, next_state):
    return np.sum(state[:92])- np.sum(next_state[:92])


def calculate_rewardV1(values, reward, total_reward):
    prev_val = np.sum(np.array(list_values[-1]))
    list_values.append(values)
    actual_val = np.sum(np.array(list_values[-1]))
    total_reward += np.sum(actual_val)
    reward = prev_val - actual_val
    return  reward, total_reward, prev_val




def calculate_reward_V2(current_state, previous_state, waiting_time_penalty):
    alpha = 0.5
    for i in range(len(current_state)):
        if current_state[i] >= previous_state[i] and previous_state[i] != 0:
            waiting_time_penalty[i] += 1
            #waiting_time_penalty[i] = (1 + 0.1) ** (waiting_time_penalty[i] + 1) * previous_state[i]
        else:
            waiting_time_penalty[i] = 0
    waiting_time_penalty = np.array(waiting_time_penalty)
    print("ffff", waiting_time_penalty)
    print(current_state * (1+alpha)**waiting_time_penalty)
    reward = np.sum(previous_state) - np.sum(current_state * (1+alpha)**waiting_time_penalty)
    #reward = np.sum(previous_state) - np.sum(current_state)
    previous_state = current_state
    # print(reward)
    return reward, waiting_time_penalty



def calculate_rewardV3(state, next_state, avg_speed, next_avg_speed, halting, next_halting):
    traffic_change = (np.sum(state[:92]) - np.sum(next_state[:92])) / (np.sum(state[:92]) + 1e-6)
    speed_change = (next_avg_speed - avg_speed) / (avg_speed + 1e-6)
    halt_penalty = (next_halting - halting) / (halting + 1e-6)
    return traffic_change + 0.5 * speed_change - 0.3 * halt_penalty
