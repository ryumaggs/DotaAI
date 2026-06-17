import time
import json
import csv
import os
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
import sys
sys.path.append(str(BASE_DIR))
import pickle
from data.data_util import load_api_key, get
from data.update_meta import *
import requests

# ── config ────────────────────────────────────────────────────────────────────
K_NEW            = 4    # number of /proMatches calls to make for NEW (recent) matches
K_OLD            = 1
USE_HERO_NAMES   = True
OUTPUT_JSON      = "pro_matches_draft.json"
OUTPUT_CSV       = "pro_matches_draft_objectives.csv"
API_KEY          = load_api_key()
RATE_LIMIT_DELAY = 0.1
# ─────────────────────────────────────────────────────────────────────────────

def load_existing(json_path):
    """
    Load existing JSON file if it exists.
    Returns (matches, seen_ids, max_match_id, min_match_id)
    max_match_id — most recent match on record (for fetching newer games)
    min_match_id — oldest match on record (for fetching older games)
    """
    if json_path is None or not os.path.exists(json_path):
        return [], set(), None, None

    with open(json_path, encoding="utf-8") as f:
        matches = json.load(f)

    if not matches:
        return [], set(), None, None

    seen_ids      = {m["match_id"] for m in matches}
    max_match_id  = max(m["match_id"] for m in matches)
    min_match_id  = min(m["match_id"] for m in matches)
    print(f"Loaded {len(matches)} existing matches.")
    print(f"  Newest on record: {max_match_id}")
    print(f"  Oldest on record: {min_match_id}")
    return matches, seen_ids, max_match_id, min_match_id


def fetch_pro_matches_newer(k, seen_ids, max_match_id, api_key = None):
    matches = []
    params = {'api_key': api_key}
    for i in range(k):
        batch = get("https://api.opendota.com/api/proMatches", params=params)
        if not batch:
            print(f"  [new] Empty response on call {i+1}, stopping.")
            break
        lowest_new_id = None
        new = 0
        overlap = False
        for m in batch:
            if m["match_id"] in seen_ids or m["match_id"] <= max_match_id:
                overlap = True
            elif m["match_id"] not in seen_ids:
                seen_ids.add(m["match_id"])
                matches.append(m)
                new += 1
            if lowest_new_id is None or m['match_id'] < lowest_new_id:
                lowest_new_id = m['match_id']

        print(f"  [new] Call {i+1}/{k}: +{new} new matches")
        print("Params: ", params)

        if overlap:
            print(f"  [new] Overlap detected, continuing")

        params['less_than_match_id'] = lowest_new_id

        time.sleep(RATE_LIMIT_DELAY)

    return matches


def fetch_pro_matches_older(k, seen_ids, min_match_id, api_key = None):
    """
    Fetch matches older than min_match_id using less_than_match_id cursor.
    """
    matches = []
    last_id = min_match_id
    for i in range(k):
        params = {"less_than_match_id": last_id, 'api_key': api_key}
        batch  = get("https://api.opendota.com/api/proMatches", params=params)
        if not batch:
            print(f"  [old] Empty response on call {i+1}, stopping.")
            break

        new = 0
        for m in batch:
            if m["match_id"] not in seen_ids:
                seen_ids.add(m["match_id"])
                matches.append(m)
                new += 1

        last_id = min(m["match_id"] for m in batch)
        print(f"  [old] Call {i+1}/{k}: +{new} matches older than {min_match_id} (cursor: {last_id})")
        time.sleep(RATE_LIMIT_DELAY)

    return matches


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


def is_valid_match(match_data):
    '''
    I have realized that pro matches can contain
    low MMR players because they p lay in some open qualifier

    This code should filter out any matches that have
    explicit players who are less than immortal rank

    If a player is unknown rank, the code is optimistic and assumes they are immortal
    '''
    players = match_data.get("players", [])
    for player in players:
        rank = player.get("rank_tier")
        if rank is None or rank == 0:
            continue  # unknown is fine
        if rank < 80:  # known rank but not immortal
            return False
    return True


def enrich(matches, id_to_name=None):
    total   = len(matches)
    valid   = []
    skipped = 0

    for i, m in enumerate(matches):
        match_id = m["match_id"]
        params = {'api_key': API_KEY}
        try:
            resp = requests.get(
                f"https://api.opendota.com/api/matches/{match_id}",
                params=params
            )
            resp.raise_for_status()
            detail = resp.json()

            if not is_valid_match(detail):
                print("Match had players less than immortal rank, skipping...")
                continue

            # --- original enrichment ---
            picks_bans = detail.get("picks_bans") or []
            if not picks_bans:
                print(f"  Skipping {match_id} — no picks_bans data.")
                skipped += 1
                continue

            picks_bans = sorted(picks_bans, key=lambda x: x.get("order", 0))

            if id_to_name:
                for entry in picks_bans:
                    entry["hero_name"] = id_to_name.get(
                        entry["hero_id"], f"hero_{entry['hero_id']}"
                    )

            m["picks_bans"] = picks_bans
            m["patch"]      = detail.get("patch", None)

            radiant_players, dire_players = [], []
            for p in detail.get("players", []):
                name = p.get("name") or p.get("personaname") or "unknown"
                if p["player_slot"] < 128:
                    radiant_players.append(name)
                else:
                    dire_players.append(name)

            m["radiant_players"] = radiant_players
            m["dire_players"]    = dire_players

            # --- objectives enrichment ---
            objectives = detail.get("objectives") or []
            parsed     = parse_objectives(objectives)
            flat       = flatten_objectives(parsed)
            m.update(flat)

            # winner from radiant_win (handles resignation case)
            m["winner"] = "goodguys" if detail.get("radiant_win") else "badguys"

            valid.append(m)

        except Exception as e:
            print(f"  Failed {match_id}: {e}")
            skipped += 1
        print(f"  Enriched {i+1}/{total} (match {match_id})")
        time.sleep(RATE_LIMIT_DELAY)

    print(f"\nEnrichment: {len(valid)} valid, {skipped} skipped.")
    return valid


def save(matches, json_path, csv_path):
    matches = sorted(matches, key=lambda x: x["match_id"], reverse=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(matches, f, indent=2)
    print(f"Saved JSON → {json_path} ({len(matches)} matches)")

    # build objective column names from flatten_objectives structure
    objective_fieldnames = [
        'fb_time', 'fb_team',
        *[f'tower_{faction}_{slot}_{field}'
          for faction in ('goodguys', 'badguys')
          for slot in ('T1_top', 'T1_mid', 'T1_bot',
                       'T2_top', 'T2_mid', 'T2_bot',
                       'T3_top', 'T3_mid', 'T3_bot', 'T4')
          for field in ('time', 'destroyed_by')],
        *[f'barracks_{faction}_{slot}_{field}'
          for faction in ('goodguys', 'badguys')
          for slot in ('melee_top', 'range_top',
                       'melee_mid', 'range_mid',
                       'melee_bot', 'range_bot')
          for field in ('time', 'destroyed_by')],
        *[f'roshan_{i}_time' for i in range(1, 4)],
        *[f'roshan_{i}_team' for i in range(1, 4)],
        'ancient_destroyed_time',
        'ancient_destroyed_loser',
        'ancient_destroyed_winner',
        'winner',
    ]

    csv_fieldnames = [
        "match_id", "patch", "league_name",
        "radiant_name", "dire_name", "radiant_win",
        "draft_order", "draft_teams", "draft_is_pick",
        "radiant_players", "dire_players",
        *objective_fieldnames,
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames, extrasaction="ignore")
        writer.writeheader()
        for m in matches:
            pb = m.get("picks_bans", [])
            row = {
                "match_id":        m.get("match_id", ""),
                "patch":           m.get("patch", ""),
                "league_name":     m.get("league_name", "") or "",
                "radiant_name":    m.get("radiant_name", "") or "",
                "dire_name":       m.get("dire_name", "") or "",
                "radiant_win":     m.get("radiant_win", ""),
                "draft_order":     ", ".join(e.get("hero_name", str(e.get("hero_id", ""))) for e in pb),
                "draft_teams":     ", ".join(str(e.get("team", ""))    for e in pb),
                "draft_is_pick":   ", ".join(str(e.get("is_pick", "")) for e in pb),
                "radiant_players": ", ".join(m.get("radiant_players", [])),
                "dire_players":    ", ".join(m.get("dire_players", [])),
            }
            # add all objective fields directly from match dict
            for field in objective_fieldnames:
                row[field] = m.get(field, None)

            writer.writerow(row)

    print(f"Saved CSV  → {csv_path}")


if __name__ == "__main__":
    try:
        with open(BASE_DIR / 'data' / 'name_id_index_maps' / 'id_to_name.pikl', 'rb') as file:
            id_to_name = pickle.load(file)
    except Exception:
        print("Error: Need to call collector_hero.py first to update hero information. Exiting...")
        exit(1)

    existing_matches, seen_ids, max_match_id, min_match_id = load_existing(OUTPUT_JSON)

    new_matches = []

    api_key = load_api_key()

    if max_match_id is not None:
        # fetch both directions
        print(f"\nFetching newer matches (match_id > {max_match_id})...")
        newer = fetch_pro_matches_newer(K_NEW, seen_ids, max_match_id)
        print(f"Found {len(newer)} newer matches.\n")

        print(f"Fetching older matches (match_id < {min_match_id})...")
        older = fetch_pro_matches_older(K_OLD, seen_ids, min_match_id)
        print(f"Found {len(older)} older matches.\n")

        new_matches = newer + older
    else:
        # first run — no existing data, just fetch normally
        print("No existing data found. Starting fresh collection...")
        last_id = None
        for i in range(K_OLD):
            params = {}
            if last_id:
                params["less_than_match_id"] = last_id
            params['api_key'] = api_key
            batch = get("https://api.opendota.com/api/proMatches", params=params)
            if not batch:
                break
            for m in batch:
                if m["match_id"] not in seen_ids:
                    seen_ids.add(m["match_id"])
                    new_matches.append(m)
            last_id = min(m["match_id"] for m in batch)
            print(f"  Call {i+1}/{K_OLD}: total so far {len(new_matches)}")
            time.sleep(RATE_LIMIT_DELAY)

    if new_matches:
        pro_matches_meta = load_meta()
        print(f"\nEnriching {len(new_matches)} collected matches...")
        new_matches = enrich(new_matches, id_to_name)
        all_matches = existing_matches + new_matches
        save(all_matches, OUTPUT_JSON, OUTPUT_CSV)
        print(f"\nDone. Total matches on record: {len(all_matches)}")
        pro_matches_meta = update_meta(pro_matches_meta, new_matches)
        save_meta(pro_matches_meta)
    else:
        print("\nNo new matches to add.")