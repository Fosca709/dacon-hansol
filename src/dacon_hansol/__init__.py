import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_PATH = Path(__file__).parents[2]
load_dotenv(ROOT_PATH / ".env")

SAVE_PATH = Path(os.environ["DACON_SAVE_PATH"])
