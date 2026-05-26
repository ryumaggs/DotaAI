import pickle
from data_util import get
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
import sys
sys.path.append(str(BASE_DIR))

def fetch_patch_map():
    patches = get("https://api.opendota.com/api/constants/patch")
    # patches is a list of {"id": int, "name": "7.xx"}
    return {p["id"]: p["name"] for p in patches}

if __name__ == "__main__":
    out = fetch_patch_map()
    with open(BASE_DIR / 'data'/ 'patch_map.pikl', 'wb') as file:
        pickle.dump(out, file)