import traci
import numpy as np
import os
from rl_package.rl_logic.annexe import calculate_reward
from rl_package.params import WINDOW

class EnvironnementSumo:
    def __init__(self, sumoCmd, window):
        """
        Initializes the SUMO environment.
        Starts SUMO with the provided command and fetches lane and traffic light IDs.
        """
        if traci.isLoaded():
            traci.close()
        traci.start(sumoCmd)
        self.window = window
        self.lanes_ids = traci.lane.getIDList()
        self.trafficlights_ids = traci.trafficlight.getIDList()
        self.position_phases = None
        self.phase_clean()

    def phase_clean(self):
        """
        Standardizes traffic light behavior for all junctions.

        - Sets each phase to a fixed duration (100s).
        - Removes automatic transitions (e.g., phase.next).
        - Filters out yellow phases and stores valid ones.

        Purpose:
        Ensures a consistent and deterministic environment across all intersections.
        This simplifies learning by making the timing of lights predictable and
        focusing agent decisions only on green/red phases, regardless of the junction type.
        """
        self.position_phases = []
        for tl in self.trafficlights_ids:
            logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl)[0]
            new_phases = [
                traci.trafficlight.Phase(duration=100, state=phase.state, minDur=100, maxDur=100)
                for phase in logic.phases
            ]
            new_logic = traci.trafficlight.Logic(
                programID=logic.programID,
                type=logic.type,
                currentPhaseIndex=logic.currentPhaseIndex,
                phases=new_phases
            )
            traci.trafficlight.setCompleteRedYellowGreenDefinition(tl, new_logic)
            self.position_phases.append(self.get_phase_without_yellow(tl)[1])


    def get_phase_without_yellow(self, traffic_light):
        """
        Returns valid phases without yellow lights for a given traffic light.
        """
        phases = traci.trafficlight.getAllProgramLogics(traffic_light)[0].phases
        long_phases, position = [], []
        for i, phase in enumerate(phases):
            if "y" not in phase.state and ("g" in phase.state or "G" in phase.state):
                long_phases.append(phase)
                position.append(i)
        return long_phases, position

    def step(self, actions):
        """
        Executes one environment step using the provided actions (phase indices).

        Returns:
            next_states (List): updated environment states
            rewards (List): rewards computed for each agent
        """
        states = [self.get_states_per_traffic_light(tl) for tl in self.trafficlights_ids]

        for i, traffic_light in enumerate(self.trafficlights_ids):
            if actions[i] < len(self.position_phases[i]):
                traci.trafficlight.setPhase(traffic_light, self.position_phases[i][actions[i]])

        for _ in range(WINDOW):
            traci.simulationStep()

        next_states = [self.get_states_per_traffic_light(tl) for tl in self.trafficlights_ids]
        rewards = [calculate_reward(states[i], next_states[i]) for i in range(len(actions))]

        return next_states, rewards

    def full_simul(self, agents):
        """
        Runs a full SUMO simulation with agent-based traffic light control.
        """
        for step in range(WINDOW*75):
            if step % WINDOW == 0:
                states = [self.get_states_per_traffic_light(tl) for tl in self.trafficlights_ids]
                actions = [agent.epsilon_greedy_policy(np.array(states[i]), 0) for i, agent in enumerate(agents)]
                for i, tl in enumerate(self.trafficlights_ids):
                    traci.trafficlight.setPhase(tl, self.position_phases[i][actions[i]])
                    
        

    def control_lanes(self, traffic_light):
        """
        Returns all unique lanes controlled by a specific traffic light.
        """
        lanes = traci.trafficlight.getControlledLanes(traffic_light)
        return list(dict.fromkeys(lanes))  # Remove duplicates

    def get_states_per_traffic_light(self, traffic_light):
        """
        Returns the normalized state representation of lanes controlled by a traffic light.
        State = [waiting time, halting number, vehicle number] per lane.
        """
        lanes = self.control_lanes(traffic_light)
        return [round(traci.lane.getWaitingTime(l)/200, 2) for l in lanes] + \
               [traci.lane.getLastStepHaltingNumber(l)/2 for l in lanes] + \
               [traci.lane.getLastStepVehicleNumber(l)/2 for l in lanes]

    def queue(self, lane_ids):
        """Returns the number of halted vehicles for a list of lanes."""
        return [traci.lane.getLastStepHaltingNumber(lane) for lane in lane_ids]

    def get_lane_no_intersection(self, lane_ids=None):
        """Filters out internal lanes (typically marked with ':')."""
        if lane_ids is None:
            lane_ids = self.lanes_ids
        return [l for l in lane_ids if not l.startswith(':')]

    def get_state(self, lane_ids):
        """Returns a basic state representation: [halting, total vehicles] per lane."""
        return [traci.lane.getLastStepHaltingNumber(l) for l in lane_ids] + \
               [traci.lane.getLastStepVehicleNumber(l) for l in lane_ids]

    def get_total_number_vehicles(self):
        """Returns the total number of vehicles in the simulation."""
        return len(traci.vehicle.getIDList())


    def close(self):
        """Closes the SUMO simulation and cleans up any active processes."""
        if traci.isLoaded():
            traci.close()
            os.system("pkill -f sumo")
            os.system("pkill -f sumo-gui")
