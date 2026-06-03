import pandas as pd
import json
import time
import requests
from pathlib import Path
from data_util import *

def parse_objectives(objectives):
    result = {
        'first_blood': None,
        'towers': {
            'goodguys': {'T1_top': None, 'T1_mid': None, 'T1_bot': None,
                         'T2_top': None, 'T2_mid': None, 'T2_bot': None,
                         'T3_top': None, 'T3_mid': None, 'T3_bot': None,
                         'T4':     None},
            'badguys':  {'T1_top': None, 'T1_mid': None, 'T1_bot': None,
                         'T2_top': None, 'T2_mid': None, 'T2_bot': None,
                         'T3_top': None, 'T3_mid': None, 'T3_bot': None,
                         'T4':     None},
        },
        'barracks': {
            'goodguys': {'melee_top': None, 'range_top': None,
                         'melee_mid': None, 'range_mid': None,
                         'melee_bot': None, 'range_bot': None},
            'badguys':  {'melee_top': None, 'range_top': None,
                         'melee_mid': None, 'range_mid': None,
                         'melee_bot': None, 'range_bot': None},
        },
        'roshan':            [],
        'ancient_destroyed': None,
    }

    TOWER_KEYS = {
        'goodguys_tower1_top': ('goodguys', 'T1_top'),
        'goodguys_tower1_mid': ('goodguys', 'T1_mid'),
        'goodguys_tower1_bot': ('goodguys', 'T1_bot'),
        'goodguys_tower2_top': ('goodguys', 'T2_top'),
        'goodguys_tower2_mid': ('goodguys', 'T2_mid'),
        'goodguys_tower2_bot': ('goodguys', 'T2_bot'),
        'goodguys_tower3_top': ('goodguys', 'T3_top'),
        'goodguys_tower3_mid': ('goodguys', 'T3_mid'),
        'goodguys_tower3_bot': ('goodguys', 'T3_bot'),
        'goodguys_tower4':     ('goodguys', 'T4'),
        'badguys_tower1_top':  ('badguys',  'T1_top'),
        'badguys_tower1_mid':  ('badguys',  'T1_mid'),
        'badguys_tower1_bot':  ('badguys',  'T1_bot'),
        'badguys_tower2_top':  ('badguys',  'T2_top'),
        'badguys_tower2_mid':  ('badguys',  'T2_mid'),
        'badguys_tower2_bot':  ('badguys',  'T2_bot'),
        'badguys_tower3_top':  ('badguys',  'T3_top'),
        'badguys_tower3_mid':  ('badguys',  'T3_mid'),
        'badguys_tower3_bot':  ('badguys',  'T3_bot'),
        'badguys_tower4':      ('badguys',  'T4'),
    }

    BARRACKS_KEYS = {
        'goodguys_melee_rax_top': ('goodguys', 'melee_top'),
        'goodguys_range_rax_top': ('goodguys', 'range_top'),
        'goodguys_melee_rax_mid': ('goodguys', 'melee_mid'),
        'goodguys_range_rax_mid': ('goodguys', 'range_mid'),
        'goodguys_melee_rax_bot': ('goodguys', 'melee_bot'),
        'goodguys_range_rax_bot': ('goodguys', 'range_bot'),
        'badguys_melee_rax_top':  ('badguys',  'melee_top'),
        'badguys_range_rax_top':  ('badguys',  'range_top'),
        'badguys_melee_rax_mid':  ('badguys',  'melee_mid'),
        'badguys_range_rax_mid':  ('badguys',  'range_mid'),
        'badguys_melee_rax_bot':  ('badguys',  'melee_bot'),
        'badguys_range_rax_bot':  ('badguys',  'range_bot'),
    }

    TEAM = {2: 'goodguys', 3: 'badguys'}

    for obj in objectives:
        t   = obj.get('time')
        typ = obj.get('type')
        key = obj.get('key', '')

        if typ == 'CHAT_MESSAGE_FIRSTBLOOD':
            slot    = obj.get('player_slot', -1)
            fb_team = 'goodguys' if slot < 128 else 'badguys'
            result['first_blood'] = {'time': t, 'team': fb_team}

        elif typ == 'building_kill':
            if 'goodguys' in key:
                destroyed = 'goodguys'
                destroyer = 'badguys'
            elif 'badguys' in key:
                destroyed = 'badguys'
                destroyer = 'goodguys'
            else:
                continue

            if 'fort' in key:
                result['ancient_destroyed'] = {
                    'time':   t,
                    'loser':  destroyed,
                    'winner': destroyer,
                }
                continue

            for substr, (faction, slot_key) in TOWER_KEYS.items():
                if substr in key:
                    result['towers'][faction][slot_key] = {
                        'time':         t,
                        'destroyed_by': destroyer,
                    }
                    break

            for substr, (faction, slot_key) in BARRACKS_KEYS.items():
                if substr in key:
                    result['barracks'][faction][slot_key] = {
                        'time':         t,
                        'destroyed_by': destroyer,
                    }
                    break

        elif typ == 'CHAT_MESSAGE_ROSHAN_KILL':
            if len(result['roshan']) < 3:
                result['roshan'].append({
                    'time': t,
                    'team': TEAM.get(obj.get('team')),
                })

    return result


def flatten_objectives(parsed):
    """Flatten parsed objectives dict into a flat dict of columns."""
    flat = {}

    # First blood
    fb = parsed['first_blood']
    flat['fb_time'] = fb['time'] if fb else None
    flat['fb_team'] = fb['team'] if fb else None

    # Towers
    for faction in ('goodguys', 'badguys'):
        for slot_key, val in parsed['towers'][faction].items():
            prefix = f'tower_{faction}_{slot_key}'
            flat[f'{prefix}_time']         = val['time']         if val else None
            flat[f'{prefix}_destroyed_by'] = val['destroyed_by'] if val else None

    # Barracks
    for faction in ('goodguys', 'badguys'):
        for slot_key, val in parsed['barracks'][faction].items():
            prefix = f'barracks_{faction}_{slot_key}'
            flat[f'{prefix}_time']         = val['time']         if val else None
            flat[f'{prefix}_destroyed_by'] = val['destroyed_by'] if val else None

    # Roshan (up to 3)
    for i in range(3):
        rosh = parsed['roshan'][i] if i < len(parsed['roshan']) else None
        flat[f'roshan_{i+1}_time'] = rosh['time'] if rosh else None
        flat[f'roshan_{i+1}_team'] = rosh['team'] if rosh else None

    # Ancient
    anc = parsed['ancient_destroyed']
    flat['ancient_destroyed_time']   = anc['time']   if anc else None
    flat['ancient_destroyed_loser']  = anc['loser']  if anc else None
    flat['ancient_destroyed_winner'] = anc['winner'] if anc else None

    return flat


def fetch_and_enrich(input_csv, output_csv, api_key = None, cache_dir='match_cache', delay=1.5):
    df        = pd.read_csv(input_csv)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    params={'api_key': api_key}
    enriched_rows = []

    for idx, row in df.iterrows():
        match_id   = row['match_id']
        cache_path = cache_dir / f"{match_id}.json"

        if cache_path.exists():
            with open(cache_path) as f:
                data = json.load(f)
            print(f"[{idx+1}/{len(df)}] {match_id} loaded from cache")
        else:
            try:
                if api_key is not None:
                    resp = requests.get(f"https://api.opendota.com/api/matches/{match_id}", params=params)
                else:
                    resp = requests.get(f"https://api.opendota.com/api/matches/{match_id}")
                resp.raise_for_status()
                data = resp.json()
                with open(cache_path, 'w') as f:
                    json.dump(data, f)
                print(f"[{idx+1}/{len(df)}] {match_id} fetched")
                time.sleep(delay)
            except Exception as e:
                print(f"[{idx+1}/{len(df)}] {match_id} FAILED: {e}")
                continue  # skip this row entirely
            
        objectives = data.get('objectives') or []
        parsed     = parse_objectives(objectives)
        flat       = flatten_objectives(parsed)
        # combine original row with new objective columns
        enriched_rows.append({**row.to_dict(), **flat})

    result_df = pd.DataFrame(enriched_rows)
    result_df.to_csv(output_csv, index=False)
    print(f"Saved {len(result_df)} rows to {output_csv}")


if __name__ == '__main__':
    fetch_and_enrich(
        input_csv  = 'pro_matches_draft.csv',
        output_csv = 'pro_matches_draft_objectives.csv',
        api_key = load_api_key(),
        cache_dir  = 'match_cache',
        delay      = 0.1,
    )