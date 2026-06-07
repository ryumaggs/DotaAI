import requests
from pathlib import Path
import pickle

def remap_hero_ids(name_to_valve_id: dict) -> dict:
    """
    Creates a new name -> tensor_index mapping that is identical to
    name -> valve_id up until the first gap, then assigns indices
    sequentially from that point forward.

    e.g. valve IDs: 1,2,3,130,132,140
         tensor idx: 1,2,3,130,131,132
    """
    # sort heroes by valve ID so we process them in order
    sorted_heroes = sorted(name_to_valve_id.items(), key=lambda x: x[1])

    name_to_tensor_idx = {}
    gap_encountered = False
    next_idx = None

    for i, (name, valve_id) in enumerate(sorted_heroes):
        if i == 0:
            name_to_tensor_idx[name] = valve_id
            next_idx = valve_id + 1
            continue

        prev_valve_id = sorted_heroes[i - 1][1]

        if not gap_encountered and valve_id == prev_valve_id + 1:
            # still sequential, mirror valve ID
            name_to_tensor_idx[name] = valve_id
            next_idx = valve_id + 1
        else:
            # first gap or already past the gap — assign next available index
            gap_encountered = True
            name_to_tensor_idx[name] = next_idx
            next_idx += 1

    return name_to_tensor_idx

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

if __name__ == "__main__":
    pass