import traci
import numpy as np
import os
from rl_package.rl_logic.annexe import calculate_reward
from rl_package.params import WINDOW
import matplotlib.pyplot as plt

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
        # print("actions",actions)
        # print("states",states)
        # print("next_states",next_states)
        n= len(actions)//5
        rewards = [calculate_reward(states[i][:n],next_states[i][:n]) for i in range(len(actions))]


        return next_states,rewards


    def full_simul(self,agents):
         ########## GLOBAL ENV STATE ##########
        global_wait_time_list = []
        global_nb_vehicules_list = []
        global_nb_halting_list = []
        global_speed_list = []
        ######################################
        ########## GLOBAL ENV STATE ##########
        global_wait_time_list = []
        global_nb_vehicules_list = []
        global_nb_halting_list = []
        global_speed_list = []
        ######################################
        for step in range(13000): ## TO CHANGED
            if step%WINDOW == 0:
                states = [self.get_states_per_traffic_light(traffic_light) for traffic_light in self.trafficlights_ids]
                actions = [agent.epsilon_greedy_policy(np.array(states[i]),0) for i,agent in enumerate(agents)]
                for i,traffic_light in enumerate(self.trafficlights_ids):
                    traci.trafficlight.setPhase(traffic_light,self.position_phases[i][actions[i]])


                ###############################################################################################
                global_wait_time = sum(traci.vehicle.getWaitingTime(veh) for veh in traci.vehicle.getIDList())
                global_nb_vehicules = len(traci.vehicle.getIDList())
                global_nb_halting = sum(1 for veh in traci.vehicle.getIDList() if traci.vehicle.getWaitingTime(veh) > 0)
                global_speed = sum(traci.vehicle.getSpeed(veh) for veh in traci.vehicle.getIDList()) / (global_nb_vehicules + 1e-10)

                global_wait_time_list.append(sum(traci.vehicle.getWaitingTime(veh) for veh in traci.vehicle.getIDList()))
                global_nb_vehicules_list.append(len(traci.vehicle.getIDList()))
                global_nb_halting_list.append(sum(1 for veh in traci.vehicle.getIDList() if traci.vehicle.getWaitingTime(veh) > 0))
                global_speed_list.append(sum(traci.vehicle.getSpeed(veh) for veh in traci.vehicle.getIDList()))
                #######################################################################################################

            traci.simulationStep()
        ########## PLOTTING ##########
        fig, axs = plt.subplots(4, 1, figsize=(10, 15))

        # Plot global wait time
        axs[0].plot(global_wait_time_list, label='Global Wait Time')
        axs[0].legend()
        axs[0].set_title('Global Wait Time')

        # Plot global number of vehicles
        axs[1].plot(global_nb_vehicules_list, label='Global Number of Vehicles')
        axs[1].legend()
        axs[1].set_title('Global Number of Vehicles')

        # Plot global number of halting vehicles
        axs[2].plot(global_nb_halting_list, label='Global Number of Halting Vehicles')
        axs[2].legend()
        axs[2].set_title('Global Number of Halting Vehicles')

        # Plot global speed
        axs[3].plot(global_speed_list, label='Global Speed')
        axs[3].legend()
        axs[3].set_title('Global Speed')

        plt.tight_layout()
        plt.show()
        ######################################

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
        #print(cleaned_lane_ids,len(cleaned_lane_ids))
        # print("\n")
        # if traffic_light ==self.trafficlights_ids[0]:
        #     print([traci.lane.getLastStepHaltingNumber(lane_id) for lane_id in cleaned_lane_ids] +\
        #     [traci.lane.getLastStepVehicleNumber(lane_id) for lane_id in cleaned_lane_ids] +\
        #     [traci.lane.getLastStepMeanSpeed(lane_id) for lane_id in cleaned_lane_ids] +\
        #     [traci.lane.getLastStepOccupancy(lane_id) for lane_id in cleaned_lane_ids] +\
        #     [traci.lane.getWaitingTime(lane_id) for lane_id in cleaned_lane_ids])
        #     print('\n')

        return [traci.lane.getLastStepHaltingNumber(lane_id) for lane_id in cleaned_lane_ids] +\
            [traci.lane.getLastStepVehicleNumber(lane_id) for lane_id in cleaned_lane_ids] +\
            [traci.lane.getLastStepMeanSpeed(lane_id) for lane_id in cleaned_lane_ids] +\
            [traci.lane.getLastStepOccupancy(lane_id) for lane_id in cleaned_lane_ids] +\
            [traci.lane.getWaitingTime(lane_id) for lane_id in cleaned_lane_ids]


    def close(self):
        if traci.isLoaded():
            traci.close()  # Properly close SUMO
            os.system("pkill -f sumo")
            os.system("pkill -f sumo-gui")
