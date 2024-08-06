import os
from dotenv import load_dotenv
from slack_bolt import App
from groq import Groq
import logging
import threading
import sys

# Add the directory containing globals.py to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("Current directory:", os.getcwd())
print("Python path:", sys.path)

from llama.globals import OS_THREAD_ID_SLACK_THREAD_MAPPER

# Load environment variables from .env.local file
load_dotenv('.env.local')

# Get the token from environment variables
slack_bot_token = os.getenv("SLACK_BOT_TOKEN")

if not slack_bot_token:
    raise ValueError("SLACK_BOT_TOKEN is not set in the environment variables")

# Initialize the Slack app with the token
slack_app = App(token=slack_bot_token)

load_dotenv()

def custom_log_factory(*args, **kwargs):
    record = base_log_factory(*args, **kwargs)

    os_thread_id = threading.get_ident()
    if type(os_thread_id) == tuple:
        os_thread_id = os_thread_id[0]

    m = OS_THREAD_ID_SLACK_THREAD_MAPPER.get(
        os_thread_id,
        {
            "channel": None,
            "thread_ts": None,
            "event_ts": None,
        },
    )
    record.slack_thread_ts = m["thread_ts"]
    record.slack_event_ts = m["event_ts"]
    record.slack_channel = m["channel"]

    return record

def create_logger(name, filename, level="INFO"):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(slack_channel)s - %(slack_thread_ts)s - %(slack_event_ts)s - %(message)s'
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if os.getenv("ENV") == "DEV":
        file_handler = logging.FileHandler(filename=filename, mode="a")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# Initialize loggers
base_log_factory = logging.getLogRecordFactory()
logging.setLogRecordFactory(custom_log_factory)

general_logger = create_logger("general", "server.log")
timeit_logger = create_logger("timeit", "timeit.log")

# Initialize clients
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))