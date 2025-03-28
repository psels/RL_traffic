import os
from dotenv import load_dotenv
load_dotenv()
# Paths to SUMO binaries, read from .env
SUMO_BIN = os.getenv("SUMO")          # Path to the SUMO binary
SUMO_GUI_BIN = os.getenv("SUMO-GUI")  # Path to the SUMO GUI binary
SIMUL_CONFIG = os.getenv("SIMUL-CONFIG")  # Path to the SUMO configuration file

# Simulation name for model saving/loading
NAME_SIMULATION = "simu_2_carrefours"  # Used as prefix when saving models

# Reinforcement Learning parameters
MEMORY_MAX_SIZE = 10000  # Max size of experience replay buffer
EPISODE = 20             # Number of training episodes
BATCH_SIZE = 32          # Training batch size
WINDOW = 200             # Number of SUMO steps between each agent decision

# Simulation parameters
SCALE = 0.5  # Vehicle spawn scaling factor (0.5 = half as many cars as original traffic
