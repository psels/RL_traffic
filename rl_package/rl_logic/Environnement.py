import traci
import numpy as np
import os
from rl_package.rl_logic.annexe import calculate_reward
from rl_package.params import WINDOW


class EnvironnementSumo:
    def __init__(self, sumoCmd,window=2000):
        if traci.isLoaded():
            traci.close()
        traci.start(sumoCmd)  # Start SUMO once
        self.window=window
        self.lanes_ids = traci.lane.getIDList()
        self.trafficlights_ids = traci.trafficlight.getIDList()
        self.position_phases = None


    def queue(self,lane_ids):
        return [traci.lane.getLastStepHaltingNumber(lane_id) for lane_id in lane_ids]

    def get_lane_no_intersection(self,lane_ids=None):
        if not lane_ids:
            lane_ids=self.lanes_ids
        return [lane_id for lane_id in lane_ids if not lane_id.startswith(':')]


    def get_state(self,lane_ids):
        return [traci.lane.getLastStepHaltingNumber(lane_id) for i,lane_id in enumerate(lane_ids) ]+\
        [traci.lane.getLastStepVehicleNumber(lane_id) for i,lane_id in enumerate(lane_ids)]

    def get_total_number_vehicles(self):
        return len(traci.vehicle.getIDList())


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


    def step(self,actions):
        ###CODER UN STEP qui prend une action en argument
        #utiliser un modele, renvoyer next state: array, reward:int, done :
        states = [self.get_states_per_traffic_light(traffic_light) for traffic_light in self.trafficlights_ids]
        for i,traffic_light in enumerate(self.trafficlights_ids):
            #traci.trafficlight.setPhase(traffic_light,2*actions[i])
            traci.trafficlight.setPhase(traffic_light,self.position_phases[i][actions[i]])

        for _ in range(self.window):
            traci.simulationStep()

        next_states = [self.get_states_per_traffic_light(traffic_light) for traffic_light in self.trafficlights_ids]
        n= len(actions)//2
        rewards = [calculate_reward(states[i][:n],next_states[i][:n]) for i in range(len(actions))]

        return next_states,rewards


    def full_simul(self,agents):
        for step in range(130000): ## TO CHANGED
            if step%WINDOW == 0:
                states = [self.get_states_per_traffic_light(traffic_light) for traffic_light in self.trafficlights_ids]
                actions = [agent.epsilon_greedy_policy(np.array(states[i]),0) for i,agent in enumerate(agents)]
                for i,traffic_light in enumerate(self.trafficlights_ids):
                    traci.trafficlight.setPhase(traffic_light,self.position_phases[i][actions[i]])
            traci.simulationStep()

    def get_number_of_junction(self):
        return traci.junction.getIDCount()


    def control_lanes(self, traffic_light):
        lane_ids = traci.trafficlight.getControlledLanes(traffic_light)

        lanes_unique = []
        for lane in lane_ids:
            if lane not in lanes_unique:
                lanes_unique.append(lane)
        return lanes_unique
        #return [lane for lane in lane_ids if not lane.startswith(':')]


    def get_states_per_traffic_light(self, traffic_light):
        lane_ids = traci.trafficlight.getControlledLanes(traffic_light)
        cleaned_lane_ids = []
        for lane in lane_ids:
            if lane not in cleaned_lane_ids:
                cleaned_lane_ids.append(lane)
        return [traci.lane.getLastStepHaltingNumber(lane_id) for lane_id in cleaned_lane_ids] +\
        [traci.lane.getLastStepVehicleNumber(lane_id) for lane_id in cleaned_lane_ids]


    def close(self):
        if traci.isLoaded():
            traci.close()  # Properly close SUMO
            os.system("pkill -f sumo")
            os.system("pkill -f sumo-gui")
