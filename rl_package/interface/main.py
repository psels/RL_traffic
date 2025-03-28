import os
import traci
import sys
import argparse
import tensorflow as tf
import numpy as np
from dotenv import load_dotenv

from rl_package.rl_logic.Environnement import EnvironnementSumo
from rl_package.rl_logic.Agent import AgentSumo
from rl_package.params import *

# Load environment variables
load_dotenv()
os.makedirs("models", exist_ok=True)


def preprocess():
    """
    Initializes a SUMO simulation environment to determine the input and output sizes
    required for each traffic light agent.

    Returns:
        inputs_per_agents (List[int]): number of input features per agent
        outputs_per_agents (List[int]): number of output actions per agent
        positions_phases (List[List[int]]): valid phase indices (excluding yellow phases)
    """
    sumoCmd = [SUMO_BIN, "-c", SIMUL_CONFIG, '--start', '--no-warnings']
    env = EnvironnementSumo(sumoCmd, WINDOW)
    inputs_per_agents = []
    outputs_per_agents = []
    positions_phases = []

    for trafficlight in env.trafficlights_ids:
        n_lanes = len(env.control_lanes(trafficlight))
        inputs_per_agents.append(n_lanes * 3)

        n_phases, position = env.get_phase_without_yellow(trafficlight)
        outputs_per_agents.append(len(n_phases))
        positions_phases.append(position)

    env.close()
    return inputs_per_agents, outputs_per_agents, positions_phases


def train_models(inputs_per_agents, outputs_per_agents, position_phases, type_model="DQN",force_new=False):
    """
    Trains one agent per traffic light and saves their models.

    Args:
        inputs_per_agents (List[int])
        outputs_per_agents (List[int])
        position_phases (List[List[int]])
        type_model (str): "DQN", "2DQN" or "3DQN"
    """
    agents = [AgentSumo(type_model, inputs, outputs) for inputs, outputs in zip(inputs_per_agents, outputs_per_agents)]

    for i, agent in enumerate(agents):
        agent.build_model()
        model_path = f"models/{NAME_SIMULATION}_{type_model}_Agent{i}.keras"
        if os.path.exists(model_path) and not force_new:
            print(f"Loading pre-trained model for Agent {i} from {model_path}...")
            agent.model_action = tf.keras.models.load_model(model_path)
            if type_model in ["2DQN", "3DQN"]:
                # 🔥 Clone proprement le modèle cible
                agent.model_target = tf.keras.models.clone_model(agent.model_action)
                agent.model_target.set_weights(agent.model_action.get_weights())

    sumoCmd = [SUMO_BIN, "-c", SIMUL_CONFIG, '--start', '--no-warnings', '--scale', str(SCALE)]

    for episode in range(EPISODE):
        print(f"Episode {episode}/{EPISODE}")
        env = EnvironnementSumo(sumoCmd, WINDOW)
        env.position_phases = position_phases
        epsilon = max(1 - episode / EPISODE, 0.01)

        states = [env.get_states_per_traffic_light(tl) for tl in env.trafficlights_ids]
        for _ in range(50):
            actions = [agent.epsilon_greedy_policy(np.array(states[i]), epsilon) for i, agent in enumerate(agents)]
            next_states, rewards = env.step(actions)

            for i in range(len(agents)):
                agents[i].add_to_memory(np.array(states[i]), np.array(actions[i]), np.array(rewards[i]), np.array(next_states[i]))

            states = next_states

            if len(agents[0].replay_buffer) >= BATCH_SIZE:
                for agent in agents:
                    agent.training_step(BATCH_SIZE)

            if env.get_total_number_vehicles() == 0:
                break

        if episode % 5 == 0 and type_model != 'DQN':
            for agent in agents:
                agent.model_target.set_weights(agent.model_action.get_weights())

        env.close()

    for i, agent in enumerate(agents):
        model_path = f"models/{NAME_SIMULATION}_{type_model}_Agent{i}.keras"
        agent.model_action.save(model_path)
        print(f"Model saved for Agent {i} at: {model_path}")


def load_trained_agents(inputs_per_agents, outputs_per_agents, type_model="DQN"):
    """
    Loads the trained agents from disk.

    Returns:
        List[AgentSumo]: list of loaded agents
    """
    agents = [AgentSumo(type_model, inputs, outputs) for inputs, outputs in zip(inputs_per_agents, outputs_per_agents)]

    for i, agent in enumerate(agents):
        model_path = f"models/{NAME_SIMULATION}_{type_model}_Agent{i}.keras"
        if os.path.exists(model_path):
            print(f"Loading pre-trained model for Agent {i} from {model_path}...")
            agent.build_model()
            agent.model_action = tf.keras.models.load_model(model_path)
        else:
            print(f"No pre-trained model found for Agent {i}.")
            sys.exit(1)

    return agents


def scenario(agents, positions_phases):
    """
    Launches a full SUMO GUI simulation using the provided agents.
    """
    sumoCmd = [SUMO_GUI_BIN, "-c", SIMUL_CONFIG, '--start', '--no-warnings', '--scale', str(SCALE)]
    env = EnvironnementSumo(sumoCmd, WINDOW)
    env.position_phases = positions_phases
    env.full_simul(agents)


def get_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser(description="Run reinforcement learning training or simulation.")
    parser.add_argument("--train", action="store_true", help="Start model training.")
    parser.add_argument("--evaluate", action="store_true", help="Run a simulation with trained models.")
    parser.add_argument("--model", type=str, choices=["DQN", "2DQN", "3DQN"], default="DQN", help="Model type")
    parser.add_argument("--fresh", action="store_true", help="Force training from scratch even if model exists.")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    type_model = args.model
    inputs_per_agents, outputs_per_agents, positions_phases = preprocess()

    print(f'Inputs per agent: {inputs_per_agents}')
    print(f'Outputs per agent: {outputs_per_agents}')
    print(f'Phase indices per agent: {positions_phases}')

    if args.train:
        train_models(inputs_per_agents, outputs_per_agents, positions_phases, type_model, force_new=args.fresh)
    elif args.evaluate:
        agents = load_trained_agents(inputs_per_agents, outputs_per_agents, type_model)
        scenario(agents, positions_phases)
    else:
        print("Specify --train to train the model or --evaluate to run a simulation.")
