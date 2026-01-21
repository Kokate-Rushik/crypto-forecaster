import os

# Absolute path of the 'src' directory
SRC_DIR = os.path.dirname(os.path.abspath(__file__))

# Absolute path of the Project Root
BASE_DIR = os.path.dirname(SRC_DIR)

# Centralized Data Paths
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

# Ensuring if folder exists
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

