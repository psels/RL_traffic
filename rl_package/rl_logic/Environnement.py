import traci
import numpy as np
import os
from rl_package.rl_logic.annexe import calculate_reward
from rl_package.params import WINDOW

class EnvironnementSumo:
    def __init__(self, sumoCmd, window):
        if traci.isLoaded():
            traci.close()
        traci.start(sumoCmd)  # Start SUMO once
        self.window = window
        self.lanes_ids = traci.lane.getIDList()
        self.trafficlights_ids = traci.trafficlight.getIDList()
        self.position_phases = None
        self.phase_clean()  # 🔥 AJOUTÉ 🔥 : Initialisation correcte des phases



    def phase_clean(self):
        """Fixe toutes les phases des feux tricolores à 30 secondes, sans transition automatique"""
        self.position_phases = []  # 🔥 Stocke les positions des phases valides (sans jaune)

        for tl in self.trafficlights_ids:
            logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl)[0]  # Récupère la logique actuelle

            # 🔁 Nouvelle liste de phases modifiées
            new_phases = []
            for phase_index, phase in enumerate(logic.phases):
                new_phase = traci.trafficlight.Phase(
                    duration=100,       # durée fixe
                    state=phase.state, # séquence des feux
                    minDur=100,         # min = max = durée fixe
                    maxDur=100
                    # ❌ plus de phase.next ici
                )
                new_phases.append(new_phase)

            # 🔁 Remplacer la logique du programme avec les phases éditées
            new_logic = traci.trafficlight.Logic(
                programID=logic.programID,
                type=logic.type,
                currentPhaseIndex=logic.currentPhaseIndex,
                phases=new_phases
            )
            traci.trafficlight.setCompleteRedYellowGreenDefinition(tl, new_logic)

            # ✅ Stocke les indices des phases sans jaune
            self.position_phases.append(self.get_phase_without_yellow(tl)[1])

        print("✅ All traffic light phases updated to fixed 30s durations.")



    # def phase_clean(self):
    #     """Fixe toutes les phases des feux tricolores à 30 secondes"""
    #     self.position_phases = []  # 🔥 AJOUTÉ 🔥 : Initialiser la liste des phases valides

    #     for tl in self.trafficlights_ids:
    #         logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl)[0]  # Récupère la config actuelle
    #         new_phases = []

    #         for phase_index, phase in enumerate(logic.phases):
    #             #print(f"Traffic Light {tl} - Phase {phase_index}: {phase.state} -> Setting duration to 30s")
    #             new_phase = traci.trafficlight.Phase(30, phase.state)  # 🔥 Fixe chaque phase à 30s
    #             new_phases.append(new_phase)

    #         # Mettre à jour la configuration du feu
    #         new_logic = traci.trafficlight.Logic(
    #             logic.programID,
    #             logic.type,
    #             logic.currentPhaseIndex,
    #             new_phases
    #         )
    #         traci.trafficlight.setCompleteRedYellowGreenDefinition(tl, new_logic)

    #         # 🔥 AJOUTÉ 🔥 : Stocker les indices des phases valides
    #         self.position_phases.append(self.get_phase_without_yellow(tl)[1])

    #     # 🔥 AJOUTÉ 🔥 : Vérification après mise à jour
    #     # for tl in self.trafficlights_ids:
    #     #     print(f"✅ Traffic Light {tl} - Active program: {traci.trafficlight.getProgram(tl)}")
    #     #     for phase in traci.trafficlight.getAllProgramLogics(tl)[0].phases:
    #     #         print(f"  - {phase.state}: {phase.duration}s")

    def get_phase_without_yellow(self, traffic_light):
        """Retourne les phases des feux tricolores sans phases jaunes"""
        phases = traci.trafficlight.getAllProgramLogics(traffic_light)[0].phases
        long_phases = []
        position = []

        for i, phase in enumerate(phases):
            if "y" not in phase.state and ("g" in phase.state or "G" in phase.state):
                long_phases.append(phase)
                position.append(i)

        return long_phases, position

    def step(self, actions):
        """Effectue une étape de simulation avec les actions choisies"""
        states = [self.get_states_per_traffic_light(traffic_light) for traffic_light in self.trafficlights_ids]

        # 🔥 AJOUTÉ 🔥 : Vérification que les actions sont valides
        for i, traffic_light in enumerate(self.trafficlights_ids):
            if actions[i] >= len(self.position_phases[i]):
                print(f"⚠️ Warning: Invalid action {actions[i]} for traffic light {traffic_light}")
            else:
                phase_index = self.position_phases[i][actions[i]]
                traci.trafficlight.setPhase(traffic_light, phase_index)

        # 🔥 AJOUTÉ 🔥 : Vérification de la mise à jour des feux
        # for j in range(self.window):
        #     if j % 5 == 0:
        #         for i, traffic_light in enumerate(self.trafficlights_ids):
        #             traci.trafficlight.setPhase(traffic_light, self.position_phases[i][actions[i]])

            traci.simulationStep()

        next_states = [self.get_states_per_traffic_light(traffic_light) for traffic_light in self.trafficlights_ids]
        rewards = [calculate_reward(states[i], next_states[i]) for i in range(len(actions))]

        return next_states, rewards

    def full_simul(self, agents):
        """Exécute une simulation complète avec les agents en mettant à jour les phases correctement"""
        for step in range(13000):  # 🔥 TO DO: Ajuste cette valeur selon la simulation requise
            if step % WINDOW == 0:
                print(f"🔄 Step {step}: Computing actions for traffic lights...")  # 🔥 Debugging log
                states = [self.get_states_per_traffic_light(traffic_light) for traffic_light in self.trafficlights_ids]
                actions = [agent.epsilon_greedy_policy(np.array(states[i]), 0) for i, agent in enumerate(agents)]

                for i, traffic_light in enumerate(self.trafficlights_ids):
                    phase_index = self.position_phases[i][actions[i]]
                    print(f"🚦 Traffic Light {traffic_light} - Applying Phase Index: {phase_index}")  # 🔥 Debugging log
                    traci.trafficlight.setPhase(traffic_light, phase_index)  # 🔥 Assurer l'application immédiate de la phase

            # 🔥 Correction : Assurer que les feux continuent à fonctionner normalement après chaque update
            # if step % 5 == 0:
            #     for i, traffic_light in enumerate(self.trafficlights_ids):
            #         phase_index = self.position_phases[i][actions[i]]
            #         traci.trafficlight.setPhase(traffic_light, phase_index)

            traci.simulationStep()  # 🔥 Exécute la simulation SUMO

    print("✅ Simulation terminée avec mise à jour correcte des phases à chaque `WINDOW` step.")

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
        return [round(traci.lane.getWaitingTime(lane_id)/200,2) for lane_id in cleaned_lane_ids] +\
            [traci.lane.getLastStepHaltingNumber(lane_id)/2 for lane_id in cleaned_lane_ids] +\
        [traci.lane.getLastStepVehicleNumber(lane_id)/2 for lane_id in cleaned_lane_ids]

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

    def close(self):
        """Ferme proprement la simulation SUMO"""
        if traci.isLoaded():
            traci.close()
            os.system("pkill -f sumo")
            os.system("pkill -f sumo-gui")
