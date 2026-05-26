import requests
def get(url, params=None, api_key=None):
    params = params or {}
    if api_key:
        params["api_key"] = api_key
    r = requests.get(url, params=params)
    r.raise_for_status()
    return r.json()