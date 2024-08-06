import requests
from utils.time import timeit

@timeit()
def get_clientell_token():
    url = "https://rev-prod-k8s.clientellone.com/clientell/api/user/login"
    body = {"email": "ruthuparna@getclientell.com", "password": "Clientell@123"}
    response = requests.post(url, json=body)
    return response.json()["access_token"]

@timeit()
def get_salesforce_token(clientell_token):
    url = "https://rev-prod-k8s.clientellone.com/api/salesforce/getAccessToken"
    headers = {"Authorization": f"Token {clientell_token}"}
    response = requests.get(url, headers=headers)
    return response.json()["access_token"]
