import os
import traci
import sys
import argparse
import tensorflow as tf
import numpy as np
from dotenv import load_dotenv


load_dotenv()

from collections import deque
from rl_package.rl_logic.Environnement import EnvironnementSumo
from rl_package.params import *
from rl_package.rl_logic.Agent import AgentSumo
sumoCmd = [SUMO_BIN, "-c",SIMUL_CONFIG,'--start','--no-warnings']

def preprocess():
    """
    Renvoie le nombre d'input ou d'output necessaire pour faire tourner le modèle"
    """
    # print(sumoCmd)
    env = EnvironnementSumo(sumoCmd,2000)
    n_lanes=len(env.get_lane_no_intersection(env.lanes_ids))
    n_outputs = len(env.get_phase_without_yellow(env.trafficlights_ids[0])[0])
    env.close()
    return n_lanes*2,n_outputs

def train(n_inputs, n_outputs,type_model="DQN"):

    agent = AgentSumo(type_model,n_inputs,n_outputs)
    agent.build_model()
    #####TRAINING
    sumoCmd= [SUMO_BIN, "-c",SIMUL_CONFIG,'--start','--no-warnings']

    #for episode in range(EPISODE):
    for episode in range(EPISODE):
        print(f'episode : {episode}')
        env=EnvironnementSumo(sumoCmd,2000)
        epsilon=max(1-episode/EPISODE,0.01)
        lane_ids = env.lanes_ids
        trafic_light_ids = env.trafficlights_ids
        state = np.array(env.get_state(env.get_lane_no_intersection()))

        action=-1
        for _ in range(50): ## TO CHANGED
            action = agent.epsilon_greedy_policy(state,epsilon)
            next_state,reward = env.step(action)
            agent.add_to_memory(state, action, reward, next_state)
            state = next_state
            if len(agent.replay_buffer) >= BATCH_SIZE*1:
                    print('entrainemnt')
                    agent.training_step(BATCH_SIZE)

            if env.get_total_number_vehicles()==0: # We stop the simulation if there is no cars left in it
                break

        if episode%5==0 and type_model!='DQN':
            agent.model_target.set_weights(agent.model_action.get_weights())
        env.close()
    return agent


def get_args():
    parser = argparse.ArgumentParser(description="Lance l'entraînement du modèle RL.")
    parser.add_argument("--model",type=str, choices=["DQN","2DQN" ,"3DQN"], default="DQN", help="Type de modèle : DQN, 2DQN, 3DQN")
    return parser.parse_args()

def scenario(agent):
    sumoCmd=[SUMO_GUI_BIN, "-c",SIMUL_CONFIG,'--start','--no-warnings']
    env=EnvironnementSumo(sumoCmd,2000)
    env.full_simul(agent)


if __name__== '__main__':
    args = get_args()
    type_model = args.model
    print(type_model)
    n_inputs,n_outputs = preprocess()
    agent = train(n_inputs,n_outputs,type_model)
    scenario(agent)
