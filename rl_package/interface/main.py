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

# SUMO command
sumoCmd = [SUMO_BIN, "-c", SIMUL_CONFIG, '--start', '--no-warnings']

def preprocess():
    """
    Determines the number of inputs and outputs required for the model.
    """
    env = EnvironnementSumo(sumoCmd, WINDOW)

    # Get the number of lanes that are not intersections
    n_lanes = len(env.get_lane_no_intersection(env.lanes_ids))

    # Get the number of valid traffic light phases (excluding yellow phases)
    n_outputs = len(env.get_phase_without_yellow(env.trafficlights_ids[0])[0])

    env.close()
    return n_lanes * 2, n_outputs  # Inputs: lane states (queue + vehicle count), Outputs: traffic light phases

def train(n_inputs, n_outputs, type_model="DQN"):
    """
    Trains a reinforcement learning model to optimize traffic lights.
    Saves the trained model after completion.
    """
    agent = AgentSumo(type_model, n_inputs, n_outputs)
    agent.build_model()
    model_path = f"models/{type_model}.keras"
    if os.path.exists(model_path):
        print(f"🔄 Loading pre-trained model {type_model}...")
        agent.model_action=tf.keras.models.load_model(model_path)
        agent.model_target=tf.keras.models.load_model(model_path)

    sumoCmd = [SUMO_BIN, "-c", SIMUL_CONFIG, '--start', '--no-warnings']

    for episode in range(EPISODE):
        print(f'🔄 Episode {episode}/{EPISODE}')

        env = EnvironnementSumo(sumoCmd, WINDOW)
        epsilon = max(1 - episode / EPISODE, 0.01)  # Decaying epsilon for exploration

        state = np.array(env.get_state(env.get_lane_no_intersection()))

        for _ in range(50):  # Steps per episode
            action = agent.epsilon_greedy_policy(state, epsilon)
            next_state, reward = env.step(action)
            agent.add_to_memory(state, action, reward, next_state)
            state = next_state

            # Train the model if there is enough experience in memory
            if len(agent.replay_buffer) >= BATCH_SIZE * 10:
                agent.training_step(BATCH_SIZE)

            # Stop the simulation if there are no vehicles left
            if env.get_total_number_vehicles() == 0:
                break

        # Update target network every 5 episodes for Double/Dueling DQN
        if episode % 5 == 0 and type_model != 'DQN':
            agent.model_target.set_weights(agent.model_action.get_weights())

        env.close()

    # Save the trained model
    model_path = f"models/{type_model}.keras"
    agent.model_action.save(model_path)
    print(f"✅ Model saved at: {model_path}")

def load_trained_agent(n_inputs, n_outputs, type_model="DQN"):
    """
    Loads a pre-trained agent from a saved model file.
    If no trained model is found, the script exits.
    """
    model_path = f"models/{type_model}.keras"

    if os.path.exists(model_path):
        print(f"🔄 Loading pre-trained model {type_model}...")
        agent = AgentSumo(type_model, n_inputs, n_outputs)
        agent.build_model()
        agent.model_action = tf.keras.models.load_model(model_path)
        return agent
    else:
        print(f"❌ No pre-trained model found for {type_model}.")
        sys.exit(1)

def scenario(agent):
    """
    Runs a SUMO simulation using the trained agent.
    """
    sumoCmd = [SUMO_GUI_BIN, "-c", SIMUL_CONFIG, '--start', '--no-warnings']
    env = EnvironnementSumo(sumoCmd, WINDOW)
    env.full_simul(agent)


def get_args():
    """
    Handles command-line arguments.
    Allows users to choose between training a model and running a scenario.
    """
    parser = argparse.ArgumentParser(description="Runs reinforcement learning training or simulation.")
    parser.add_argument("--train", action="store_true", help="Start model training.")
    parser.add_argument("--evaluate", action="store_true", help="Run a simulation with a trained model.")
    parser.add_argument("--model", type=str, choices=["DQN", "2DQN", "3DQN"], default="DQN",
                        help="Model type: DQN, 2DQN, or 3DQN")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    type_model = args.model
    n_inputs, n_outputs = preprocess()

    if args.train:
        train(n_inputs, n_outputs, type_model)
    elif args.evaluate:
        agent = load_trained_agent(n_inputs, n_outputs, type_model)
        scenario(agent)
    else:
        print("❌ Specify --train to train the model or --evaluate to run a simulation.")
