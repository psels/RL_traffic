import os
from dotenv import load_dotenv
load_dotenv()

SUMO_BIN = os.getenv("SUMO")
SUMO_GUI_BIN = os.getenv("SUMO-GUI")
SIMUL_CONFIG = os.getenv("SIMUL-CONFIG")
NAME_SIMULATION = "simu_simple"
MEMORY_MAX_SIZE = 10000
EPISODE = 5
BATCH_SIZE = 32
WINDOW=200
