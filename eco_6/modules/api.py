"""
API, one class per service/endpoint
-
dev:
post, put, delete
api key
auth token, crypto

see /examples/
"""
import requests

class Endpoint:
    # --------------------------- init ---------------------------
    def __init__(self, URL:str, API_KEY:str=None, params=None):
        self.URL = URL
        self.API_KEY = API_KEY
        self.params = params
        
    def setParams(self, params):
        self.params = params
    
    def get(self):
        res = requests.get(self.URL)
        self.status = res.status_code
        return res.json()