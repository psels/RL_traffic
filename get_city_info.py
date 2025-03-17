########### IMPORTS ###########
import os
import traci
from dotenv import load_dotenv
import numpy as np
import time
###############################



########## ENVIRONMENT VARIABLES ##########
load_dotenv()

sumo_bin = os.getenv("SUMO")
sumo_gui_bin = os.getenv("SUMO-GUI")
simulConfig = "city_traffic_V2/city2.sumocfg"
#simulConfig = os.getenv("SIMUL-CONFIG2")
###########################################



########## DEBUG MODE ##########
## 0: No Debugging ## 1: Debugging important print ## 2: Debugging full print
debugg_mode = 1
################################


########## REALISTIC MODE ##########
realistic_distance = 250
realistic_mode = False
####################################



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
lane_ids =  traci.lane.getIDList()

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



########## GETTING LANE FOR EACH TRAFFIC LIGHT IDS ##########
if realistic_mode == False:
    print(f"#### Realistic DESACTIVATED")

    nb_of_traffic_light_ids = len(traffic_light_ids)

    if debugg_mode >= 2:
        print(f"#### Nb of traffic light ids = {nb_of_traffic_light_ids}\n\n\n")

    traffic_lights_dict = {}

    for traffic_light_id in traffic_light_ids:
        controlled_lanes = traci.trafficlight.getControlledLanes(traffic_light_id)
        lane_halting = {}
        lane_occupency = {}
        lane_speed = {}

        for lane_id in controlled_lanes:
            lane_halting[lane_id] = traci.lane.getLastStepHaltingNumber(lane_id)
            lane_occupency[lane_id] = traci.lane.getLastStepHaltingNumber(lane_id)
            lane_speed[lane_id] = traci.lane.getLastStepMeanSpeed(lane_id)

        traffic_lights_dict[traffic_light_id] = {
            "nb_lanes": len(controlled_lanes),
            "lanes_list": controlled_lanes,
            "lane_halting": lane_halting,
            "lane_occupency": lane_occupency,
            "lane_speed": lane_speed
        }

    if debugg_mode >= 1:
        print(f"#### Traffic Lights Dictionary = \n{traffic_lights_dict[traffic_light_ids[0]]}\n\n\n")


if realistic_mode == True:
    print(f"#### Realistic ACTIVATED meters = {realistic_distance}")

    nb_of_traffic_light_ids = len(traffic_light_ids)

    if debugg_mode >= 2:
        print(f"#### Nb of traffic light ids = {nb_of_traffic_light_ids}\n\n\n")

    traffic_lights_dict = {}

    for traffic_light_id in traffic_light_ids:
        controlled_lanes = traci.trafficlight.getControlledLanes(traffic_light_id)
        lane_halting = {}
        lane_occupency = {}
        lane_speed = {}


        for lane_id in controlled_lanes:
            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)  # Récupère les véhicules sur la lane

        # Filtrer uniquement les véhicules situés à moins de 250m
        vehicles_nearby = [
            veh_id for veh_id in vehicle_ids if traci.vehicle.getLanePosition(veh_id) <= realistic_distance
        ]

        lane_occupency[lane_id] = len(vehicles_nearby)  # Nombre total de véhicules <= realistic_distance

        # Moyenne de vitesse des véhicules dans les 250m (évite la division par 0)
        lane_speed[lane_id] = (
            sum(traci.vehicle.getSpeed(veh_id) for veh_id in vehicles_nearby) / len(vehicles_nearby)
            if vehicles_nearby else 0
        )

        traffic_lights_dict[traffic_light_id] = {
            "nb_lanes": len(controlled_lanes),
            "lanes_list": controlled_lanes,
            "lane_occupency": lane_occupency,
            "lane_speed": lane_speed
        }

    if debugg_mode >= 1:
        print(f"#### Traffic Lights Dictionary = \n{traffic_lights_dict[traffic_light_ids[0]]}\n\n\n")
################################################################




########## GLOBAL ENV STATE ##########
global_wait_time = sum(traci.vehicle.getWaitingTime(veh) for veh in traci.vehicle.getIDList())
global_nb_veh = len(traci.vehicle.getIDList())
global_nb_halting = sum(traci.vehicle.getAccumulatedWaitingTime(veh) for veh in traci.vehicle.getIDList())
global_speed = sum(traci.vehicle.getSpeed(veh) for veh in traci.vehicle.getIDList())

if debugg_mode >= 1:
    print(f"#### Global Wait Time = {global_wait_time}")
    print(f"#### Global Nb Veh = {global_nb_veh}")
    print(f"#### Global Nb Halting = {global_nb_halting}")
    print(f"#### Global Speed = {global_speed}")
######################################




simulation_end_time = 1440

# Lancer la simulation et vérifier si la simulation a atteint le temps de fin.
while traci.simulation.getTime() < simulation_end_time:
    traci.simulationStep()  # Effectuer un pas de simulation
    time.sleep(0.1)  # Ajoutez un délai pour ne pas trop solliciter le processeur (facultatif)

# la simulation a atteint l'heure de fin
print("le temps de simulation est terminé. Fermeture de la simulation.")
traci.close()
