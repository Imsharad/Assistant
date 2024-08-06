from simple_salesforce import Salesforce
from utils.auth import get_clientell_token, get_salesforce_token
import cachetools.func
from utils.time import timeit
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

@timeit()
@cachetools.func.ttl_cache(maxsize=1, ttl=600)
def get_salesforce_connection():
    clientell_token = get_clientell_token()
    salesforce_token = get_salesforce_token(clientell_token)
    sf = Salesforce(
        instance_url=os.getenv('SALESFORCE_INSTANCE_URL'),
        session_id=salesforce_token,
    )
    return sf
