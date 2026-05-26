import pickle
from data_util import get
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
import sys
sys.path.append(str(BASE_DIR))

from data_util import get
def fetch_hero_map():
    heroes = get("https://api.opendota.com/api/constants/heroes")
    id_to_name = {v["id"]: v["localized_name"] for v in heroes.values()}
    name_to_id = {v["localized_name"]: v["id"] for v in heroes.values()}
    return id_to_name, name_to_id


if __name__ == "__main__":
    id_to_name, name_to_id = fetch_hero_map()
    with open(BASE_DIR / 'data'/ 'id_to_name.pikl', 'wb') as file:
        pickle.dump(id_to_name, file)
    
    with open(BASE_DIR / 'data'/ 'name_to_id.pikl', 'wb') as file:
        pickle.dump(name_to_id, file)
