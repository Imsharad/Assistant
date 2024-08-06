import requests
from utils.time import timeit
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

@timeit()
def get_clientell_token():
    url = os.getenv('CLIENTELL_LOGIN_URL')
    email = os.getenv('CLIENTELL_EMAIL')
    password = os.getenv('CLIENTELL_PASSWORD')
    body = {"email": email, "password": password}
    response = requests.post(url, json=body)
    return response.json()["access_token"]

@timeit()
def get_salesforce_token(clientell_token):
    url = os.getenv('SALESFORCE_TOKEN_URL')
    headers = {"Authorization": f"Token {clientell_token}"}
    response = requests.get(url, headers=headers)
    return response.json()["access_token"]


    CLIENTELL_LOGIN_URL = "https://rev-prod-k8s.clientellone.com/clientell/api/user/login"
    CLIENTELL_EMAIL = {"email": "ruthuparna@getclientell.com", "CLIENTELL_PASSWORD": "Clientell@123"}
    SALESFORCE_TOKEN_URL = "https://rev-prod-k8s.clientellone.com/api/salesforce/getAccessToken"
