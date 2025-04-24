# === Logging Setup ===
import logging


LOG_FILE = 'app.log'
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_info(message):
    print(message)
    logging.info(message)

def log_error(message):
    print(f"Error {message}")
    logging.error(message)

def log_warning(message):
    print(f"Warning {message}")
    logging.warning(message)

