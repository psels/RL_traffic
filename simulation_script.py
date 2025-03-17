########### IMPORTS ###########
import os
import traci
from dotenv import load_dotenv
import numpy as np
import time
import matplotlib.pyplot as plt
###############################



########## ENVIRONMENT VARIABLES ##########
load_dotenv()

sumo_bin = os.getenv("SUMO")
sumo_gui_bin = os.getenv("SUMO-GUI")
simulConfig = "city_traffic_long/city.sumocfg"
###########################################



########## DEBUG MODE ##########
## 0: No Debugging ## 1: Debugging important print ## 2: Debugging full print
debugg_mode = 1
################################


########## REALISTIC MODE ##########
realistic_distance = 250
realistic_mode = False
nb_step = 0
####################################


########## GLOBAL ENV STATE ##########
global_wait_time_list = []
global_nb_vehicules_list = []
global_nb_halting_list = []
global_speed_list = []
######################################


########## DEFINITION SUMO CMD ##########
seed_value = 1

sumoCmd = [sumo_gui_bin, "-c",
           simulConfig, "--start",
           #"--no-warnings",
           "--seed", str(seed_value),
           '--scale', str(np.random.uniform(0.1, 0.2))]
#########################################



########## STARTING SIMULATION ##########
if traci.isLoaded():
    traci.close()

traci.start(sumoCmd)
#########################################



########## GETTING LANE AND TRAFFIC LIGHT IDS ##########
lane_ids = traci.lane.getIDList()

if debugg_mode >= 2:
    print(f"#### Lane_ids = \n{lane_ids}\n\n\n")

traffic_light_ids = traci.trafficlight.getIDList()

if debugg_mode >= 2:
    print(f"#### Traffic_light_ids = \n{traffic_light_ids}\n\n\n")
########################################################



########## STATE PROCESSING ##########
cleaned_lane_ids = [lane for lane in lane_ids if not lane.startswith(':')]

if debugg_mode >= 2:
    print(f"#### Cleaned_lane_ids = \n{cleaned_lane_ids}\n\n\n")
#######################################



########## SIMULATION LOOP ##########
simulation_end_time = 1440

while traci.simulation.getTime() < simulation_end_time:
    traci.simulationStep()
    nb_step += 1

    #Actualiser le dictionnaire des feux de circulation à chaque step
    traffic_lights_dict = {}

    if realistic_mode == False:
        print(f"#### Realistic DESACTIVATED")

        nb_of_traffic_light_ids = len(traffic_light_ids)

        if debugg_mode >= 2:
            print(f"#### Nb of traffic light ids = {nb_of_traffic_light_ids}\n\n\n")

        for traffic_light_id in traffic_light_ids:
            controlled_lanes = traci.trafficlight.getControlledLanes(traffic_light_id)
            lane_halting = {}
            lane_occupency = {}
            lane_speed = {}

            for lane_id in controlled_lanes:
                lane_halting[lane_id] = traci.lane.getLastStepHaltingNumber(lane_id)
                lane_occupency[lane_id] = traci.lane.getLastStepOccupancy(lane_id)
                lane_speed[lane_id] = traci.lane.getLastStepMeanSpeed(lane_id)

            traffic_lights_dict[traffic_light_id] = {
                "nb_lanes": len(controlled_lanes),
                "lanes_list": controlled_lanes,
                "lane_halting": lane_halting,
                "lane_occupency": lane_occupency,
                "lane_speed": lane_speed
            }

        if debugg_mode >= 1:
            print(f"#### Traffic Lights Dictionary = \n{traffic_lights_dict}\n\n\n")
            #print(f"#### junction 3 occupency = \n{traffic_lights_dict['junction_6']['lane_occupency']}\n\n\n")
            #print(f"#### junction 3 halting = \n{traffic_lights_dict['junction_6']['lane_halting']}\n\n\n")
            for keys, values in traffic_lights_dict['junction_1'].items():
                print(f"#### {keys}  = \n{values}\n")
                #print(f"#### {keys}  = \n{values['lane_halting']}\n\n\n")
    if realistic_mode == True:
        print(f"#### Realistic ACTIVATED meters = {realistic_distance}")

        vehicle_type_passenger = {
            'bus': 30,
            'cautious': 1.3,
            'aggressive': 1.3,
            'truck': 1,
            'motorcycle': 1.1
        }

        nb_of_traffic_light_ids = len(traffic_light_ids)

        if debugg_mode >= 2:
            print(f"#### Nb of traffic light ids = {nb_of_traffic_light_ids}\n\n\n")

        for traffic_light_id in traffic_light_ids:
            controlled_lanes = traci.trafficlight.getControlledLanes(traffic_light_id)
            current_phase = traci.trafficlight.getPhase(traffic_light_id)
            all_phases = str(traci.trafficlight.getCompleteRedYellowGreenDefinition(traffic_light_id))
            lane_halting = {}
            passenger_halting = {}
            passenger_wait_time = {}
            lane_occupency = {}
            lane_speed = {}

            for lane_id in controlled_lanes:
                vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)  # Récupère les véhicules sur la lane

                # Filtrer uniquement les véhicules situés à moins de realistic_distance
                vehicles_nearby = [
                    veh_id for veh_id in vehicle_ids if traci.vehicle.getLanePosition(veh_id) <= realistic_distance
                ]

                # Initialisation des compteurs
                halted_count = 0
                passenger_halted = 0.0
                total_passenger_wait_time = 0.0

                for veh_id in vehicles_nearby:
                    if traci.vehicle.getSpeed(veh_id) < 0.1:
                        halted_count += 1

                        # Calcul des pondérations
                        v_type = traci.vehicle.getTypeID(veh_id)
                        passenger = vehicle_type_passenger.get(v_type, 1.0)

                        passenger_halted += passenger
                        total_passenger_wait_time += traci.vehicle.getWaitingTime(veh_id) * passenger
                        print('ici')

                lane_halting[lane_id] = halted_count
                passenger_halting[lane_id] = passenger_halted
                passenger_wait_time[lane_id] = total_passenger_wait_time
                lane_occupency[lane_id] = len(vehicles_nearby)
                lane_speed[lane_id] = (sum(traci.vehicle.getSpeed(veh_id) for veh_id in vehicles_nearby) / (len(vehicles_nearby) + 1e-10))


            traffic_lights_dict[traffic_light_id] = {
                "all_phases" : {all_phases},
                "current_phase": {current_phase},
                "nb_lanes": len(controlled_lanes),
                "lanes_list": controlled_lanes,
                "lane_halting": lane_halting,
                "passenger_halting": passenger_halting,
                "passenger_wait_time": passenger_wait_time,
                "lane_occupency": lane_occupency,
                "lane_speed": lane_speed
            }

        if debugg_mode >= 1:
            #print(f"#### Traffic Lights Dictionary = \n{traffic_lights_dict}\n\n\n")
            #print(f"#### junction_6 passenger_halting = \n{traffic_lights_dict['junction_6']['passenger_halting']}\n\n\n")
            print(f"#### junction_6 passenger_wait_time = \n{traffic_lights_dict['junction_6']['passenger_wait_time']}\n\n\n")

    ########## GLOBAL ENV STATE ##########
    global_wait_time = sum(traci.vehicle.getWaitingTime(veh) for veh in traci.vehicle.getIDList())
    global_nb_vehicules = len(traci.vehicle.getIDList())
    global_nb_halting = sum(1 for veh in traci.vehicle.getIDList() if traci.vehicle.getWaitingTime(veh) > 0)
    global_speed = sum(traci.vehicle.getSpeed(veh) for veh in traci.vehicle.getIDList()) / (global_nb_vehicules + 1e-10)


    global_wait_time_list.append(sum(traci.vehicle.getWaitingTime(veh) for veh in traci.vehicle.getIDList()))
    global_nb_vehicules_list.append(len(traci.vehicle.getIDList()))
    global_nb_halting_list.append(sum(1 for veh in traci.vehicle.getIDList() if traci.vehicle.getWaitingTime(veh) > 0))
    global_speed_list.append(sum(traci.vehicle.getSpeed(veh) for veh in traci.vehicle.getIDList()))

    if debugg_mode >= 1:
        print(f"#### Global Wait Time = {global_wait_time}")
        print(f"#### Global Nb Veh = {global_nb_vehicules}")
        print(f"#### Global Nb Halting = {global_nb_halting}")
        print(f"#### Global Speed = {global_speed}")
        print(f"#### Step = {nb_step}")
        print('\n\n\n')


# la simulation a atteint l'heure de fin
print("Le temps de simulation est terminé. Fermeture de la simulation.")
traci.close()



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
##############################
