from pathlib import Path

# Define the base directory for project files
BASE_DIR = Path(__file__).parent.parent.parent

# Define the data directory
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / '.outputs'
LOGS_DIR = BASE_DIR / '.logs'


if __name__ == "__main__":
	None