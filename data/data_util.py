import requests
from pathlib import Path
def get(url, params=None, api_key=None):
    params = params or {}
    if api_key:
        params["api_key"] = api_key
    r = requests.get(url, params=params)
    r.raise_for_status()
    return r.json()

def load_api_key():
    key_path = Path(__file__).parent.parent / 'secrets' / 'api_key.txt'
    with open(key_path, 'r') as f:
        key = f.read().strip()
    return key