import logging
import os
from datetime import datetime

LOG_DIR = "logs"
CURRENT_TIME_STAMP = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
LOG_FIlE_NAME = f"log_{CURRENT_TIME_STAMP}.log"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FIlE_PATH = os.path.join(LOG_DIR, LOG_FIlE_NAME)

logging.basicConfig(filename=LOG_FIlE_PATH,
                    filemode="w",
                    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger("textSummarizerLogger")
