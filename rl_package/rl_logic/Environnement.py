import traci
import numpy as np
import os
from rl_package.rl_logic.annexe import calculate_reward

class EnvironnementSumo:
    def __init__(self, sumoCmd,window=2000):
        if traci.isLoaded():
            traci.close()
        traci.start(sumoCmd)  # Start SUMO once
        self.window=window
        self.lanes_ids = traci.lane.getIDList()
        self.trafficlights_ids = traci.trafficlight.getIDList()


    def queue(self,lane_ids):
        return [traci.lane.getLastStepHaltingNumber(lane_id) for lane_id in lane_ids]

    def get_lane_no_intersection(self,lane_ids=None):
        if not lane_ids:
            lane_ids=self.lanes_ids
        return [lane_id for lane_id in lane_ids if not lane_id.startswith(':')]


    def get_state(self,lane_ids):
        return [traci.lane.getLastStepHaltingNumber(lane_id) for i,lane_id in enumerate(lane_ids) ]+\
        [traci.lane.getLastStepVehicleNumber(lane_id) for i,lane_id in enumerate(lane_ids)]


    def get_phase_without_yellow(self,traffic_light):
        "return phases of trafific_light without yellow phase"
        phases = traci.trafficlight.getAllProgramLogics(traffic_light)[0].phases
        long_phases = []
        position = []
        for i,phase in enumerate(phases):
            if "y" not in phase.state:
                long_phases.append(phase)
                position.append(i)
        return long_phases, position


    def step(self,action):
        ###CODER UN STEP qui prend une action en argument
        #utiliser un modele, renvoyer next state: array, reward:int, done :
        lanes = self.get_lane_no_intersection()
        state = np.array(self.get_state(lanes))
        traci.trafficlight.setPhase(self.trafficlights_ids[0],2*action)

        for _ in range(self.window):
            traci.simulationStep()

        next_state = np.array(self.get_state(lanes))
        reward = calculate_reward(state,next_state)

        return next_state,reward

    def full_simul(self,agent):
        lanes = self.get_lane_no_intersection()
        state = np.array(self.get_state(lanes))
        action=1
        for step in range(130000): ## TO CHANGED
            if step%2000 == 0:
                state=np.array(self.get_state(lanes))
                action = agent.epsilon_greedy_policy(state,0)*2
                traci.trafficlight.setPhase(self.trafficlights_ids[0],action)
            traci.simulationStep()



    def close(self):
        if traci.isLoaded():
            traci.close()  # Properly close SUMO
            os.system("pkill -f sumo")
            os.system("pkill -f sumo-gui")
