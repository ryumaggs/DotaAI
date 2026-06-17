import requests
import pandas as pd
from data_util import load_api_key, get
import time
from tqdm import tqdm

API_KEY          = load_api_key()
RATE_LIMIT_DELAY = 0.1
def fetch_match(match_id):
    params = {'api_key': API_KEY}
    url = f"https://api.opendota.com/api/matches/{match_id}"
    return requests.get(url, params=params).json()

def is_valid_match(match_data):
    players = match_data.get("players", [])
    for player in players:
        rank = player.get("rank_tier")
        if rank is None:
            continue  # unknown is fine
        if rank < 80:  # known rank but not immortal
            return False
    return True

if __name__ == "__main__":
    df = pd.read_csv('./pro_matches_draft_objectives.csv')
    valid_indices = []
    invalid_indices = []
    counter = 0
    for idx, row in tqdm(df.iterrows(),total=len(df)):
        match_data = fetch_match(row['match_id'])
        if is_valid_match(match_data):
            valid_indices.append(idx)
        else:
            invalid_indices.append(idx)
        if counter % 100 == 0:
            tdf = df.loc[valid_indices].reset_index(drop=True)
            tdf.to_csv("./pro_filtered_matches.csv", index=False)
            fdf = df.loc[invalid_indices].reset_index(drop=True)
            tdf.to_csv("./non_pro_filtered_matches.csv", index=False)
        counter += 1
        time.sleep(RATE_LIMIT_DELAY)
    tdf = df.loc[valid_indices].reset_index(drop=True)
    tdf.to_csv("./pro_filtered_matches.csv", index=False)
    fdf = df.loc[invalid_indices].reset_index(drop=True)
    tdf.to_csv("./non_pro_filtered_matches.csv", index=False)
