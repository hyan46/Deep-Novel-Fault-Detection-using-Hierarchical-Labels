from dotenv import load_dotenv
from pathlib import Path
import os
import pdb
print(os.getcwd())
env_path = Path.cwd()/'.env'
load_dotenv(dotenv_path=env_path)
# ROLLING_DATA_PATH = os.path.abspath(os.getenv("ROLLING_DATA_PATH"))
# assert Path(ROLLING_DATA_PATH).exists()
ROLLING_DATA_PATH=''
CHECKPOINT_DIR = os.path.abspath(os.getenv("CHECKPOINT_DIR"))
assert Path(CHECKPOINT_DIR).exists()
ARTIFACTS_DIR = os.path.abspath(os.getenv("ARTIFACTS_DIR"))

DERIVATIVES_DIR = os.path.abspath(os.getenv("DERIVATIVES_DIR"))
print('DERIVATIVES_DIR',DERIVATIVES_DIR)
REPORT_DIR =os.path.abspath(os.getenv("REPORT_DIR"))
N_PARALLEL = 4
assert isinstance(N_PARALLEL, int) and N_PARALLEL > 0
