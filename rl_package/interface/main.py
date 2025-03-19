import os
import traci
import sys
import argparse
import tensorflow as tf
import numpy as np
from dotenv import load_dotenv

# Import internal modules
from rl_package.rl_logic.Environnement import EnvironnementSumo
from rl_package.rl_logic.Agent import AgentSumo
from rl_package.params import *


# Load environment variables
load_dotenv()

os.makedirs("models", exist_ok=True)
# SUMO command
sumoCmd = [SUMO_BIN, "-c", SIMUL_CONFIG, '--start', '--no-warnings']

def preprocess():
    """
    Determines the number of inputs and outputs required for each agent.
    """
    sumoCmd = [SUMO_BIN, "-c", SIMUL_CONFIG, '--start', '--no-warnings']
    env = EnvironnementSumo(sumoCmd, WINDOW)
    inputs_per_agents = []
    outputs_per_agents = []
    positions_phases = []

    for trafficlight in env.trafficlights_ids:
        # Get the number of lanes controlled by this traffic light
        n_lanes = len(env.control_lanes(trafficlight))
        inputs_per_agents.append(n_lanes * 2)  # Inputs: queue + vehicle count

        # Get the number of valid traffic light phases (excluding yellow)
        n_phases,position = env.get_phase_without_yellow(trafficlight)
        #print(f'trafficlight :{trafficlight},\n lane associated {env.control_lanes(trafficlight)}')
        n_outputs = len(n_phases)
        positions_phases.append(position)
        outputs_per_agents.append(n_outputs)

    env.close()
    return inputs_per_agents, outputs_per_agents,positions_phases  # List of inputs, outputs per agent, and the postion phases of each trafficlight


def train_models(inputs_per_agents, outputs_per_agents, position_phases, type_model="DQN"):
    """
    Trains multiple reinforcement learning agents to optimize traffic lights.
    Saves each model separately.
    """
    agents = [AgentSumo(type_model, inputs, outputs) for inputs, outputs in zip(inputs_per_agents, outputs_per_agents)]

    # Load pre-trained models if available
    for i, agent in enumerate(agents):
        agent.build_model()
        model_path = f"models/{NAME_SIMULATION}_{type_model}_Agent{i}.keras"
        if os.path.exists(model_path):
            print(f"🔄 Loading pre-trained model for Agent {i} from {model_path}...")
            agent.model_action = tf.keras.models.load_model(model_path)
            if agent.model_target:  # For Double/Dueling DQN
                agent.model_target = tf.keras.models.load_model(model_path)






    for episode in range(EPISODE):
        sumoCmd = [SUMO_BIN, "-c", SIMUL_CONFIG, '--start', '--no-warnings', '--seed', str(episode),  '--scale', str(np.random.uniform(0.1, 0.5))]
        print(f'🔄 Episode {episode}/{EPISODE}')
        env = EnvironnementSumo(sumoCmd, WINDOW)

        #Store the position phases of the trafficlight in the environment
        env.position_phases = positions_phases

        epsilon = max(1 - episode / EPISODE, 0.01)  # Decaying epsilon for exploration
        #epsilon = max(1 - episode +nb d'episode déja passé/ 50, 0.01)  # Decaying epsilon for exploration

        traffic_lights = env.trafficlights_ids
        states = [env.get_states_per_traffic_light(traffic_light) for traffic_light in traffic_lights]
        print(states[0])

        for _ in range(150):  # Steps per episode
            actions = [agent.epsilon_greedy_policy(np.array(states[i]), epsilon) for i, agent in enumerate(agents)]
            next_states, rewards = env.step(actions)

            for i in range(len(agents)):
                agents[i].add_to_memory(np.array(states[i]), np.array(actions[i]), np.array(rewards[i]), np.array(next_states[i]))

            states = next_states

            if len(agents[0].replay_buffer) >= BATCH_SIZE * (EPISODE/10):
                for agent in agents:
                    agent.training_step(BATCH_SIZE)

            if env.get_total_number_vehicles() == 0:
                break  # Stop simulation if no vehicles left

        # Update target networks every 5 episodes for Double/Dueling DQN
        if episode % 5 == 0 and type_model != 'DQN':
            for agent in agents:
                agent.model_target.set_weights(agent.model_action.get_weights())

        env.close()

    for i, agent in enumerate(agents):
        model_path = f"models/{NAME_SIMULATION}_{type_model}_Agent{i}.keras"
        agent.model_action.save(model_path)
        print(f"✅ Model saved for Agent {i} at: {model_path}")


def load_trained_agents(inputs_per_agents, outputs_per_agents, type_model="DQN"):
    """
    Loads pre-trained agents from saved model files.
    If any model is missing, exits the program.
    """
    agents = [AgentSumo(type_model, inputs, outputs) for inputs, outputs in zip(inputs_per_agents, outputs_per_agents)]

    for i, agent in enumerate(agents):
        model_path = f"models/{NAME_SIMULATION}_{type_model}_Agent{i}.keras"
        if os.path.exists(model_path):
            print(f"🔄 Loading pre-trained model for Agent {i} from {model_path}...")
            agent.build_model()
            agent.model_action = tf.keras.models.load_model(model_path)
        else:
            print(f"❌ No pre-trained model found for Agent {i}.")
            sys.exit(1)

    return agents

def scenario(agents,positions_phases):
    """
    Runs a SUMO simulation using the trained agents.
    """

    sumoCmd = [SUMO_GUI_BIN, "-c", SIMUL_CONFIG, '--start', '--no-warnings', '--scale', '0.5']
    env = EnvironnementSumo(sumoCmd, WINDOW)
    #Store the position phases of the trafficlight in the environment
    env.position_phases = positions_phases
    env.full_simul(agents)


def get_args():
    """
    Handles command-line arguments.
    Allows users to choose between training a model and running a scenario.
    """
    parser = argparse.ArgumentParser(description="Runs reinforcement learning training or simulation.")
    parser.add_argument("--train", action="store_true", help="Start model training.")
    parser.add_argument("--evaluate", action="store_true", help="Run a simulation with a trained models.")
    parser.add_argument("--model", type=str, choices=["DQN", "2DQN", "3DQN"], default="DQN",
                        help="Model type: DQN, 2DQN, or 3DQN")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    type_model = args.model
    inputs_per_agents, outputs_per_agents,positions_phases = preprocess()
    print(f'inputs : {inputs_per_agents}')
    print(f'outputs : {outputs_per_agents}')
    print(f'positions : {positions_phases}')
    if args.train:
        train_models(inputs_per_agents, outputs_per_agents, positions_phases, type_model)
    elif args.evaluate:
        agents = load_trained_agents(inputs_per_agents, outputs_per_agents, type_model)
        scenario(agents,positions_phases)
    else:
        print("❌ Specify --train to train the model or --evaluate to run a simulation.")
