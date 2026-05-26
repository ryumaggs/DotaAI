import time
import json
import csv
import os
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
import sys
sys.path.append(str(BASE_DIR))
import pickle
from data.data_util import get
from data.update_meta import *

# ── config ────────────────────────────────────────────────────────────────────
K_NEW            = 0    # number of /proMatches calls to make for NEW (recent) matches
K_OLD            = 5    # number of /proMatches calls to make for OLD (historical) matches
USE_HERO_NAMES   = True
OUTPUT_JSON      = "pro_matches_draft.json"
OUTPUT_CSV       = "pro_matches_draft.csv"
API_KEY          = None
RATE_LIMIT_DELAY = 1.1
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


def fetch_pro_matches_newer(k, seen_ids, max_match_id):
    matches = []

    for i in range(k):
        batch = get("https://api.opendota.com/api/proMatches")
        if not batch:
            print(f"  [new] Empty response on call {i+1}, stopping.")
            break

        new = 0
        overlap = False
        for m in batch:
            if m["match_id"] in seen_ids or m["match_id"] <= max_match_id:
                overlap = True
            elif m["match_id"] not in seen_ids:
                seen_ids.add(m["match_id"])
                matches.append(m)
                new += 1

        print(f"  [new] Call {i+1}/{k}: +{new} new matches")

        if overlap:
            print(f"  [new] Overlap detected, stopping.")
            break

        time.sleep(RATE_LIMIT_DELAY)

    return matches


def fetch_pro_matches_older(k, seen_ids, min_match_id):
    """
    Fetch matches older than min_match_id using less_than_match_id cursor.
    """
    matches = []
    last_id = min_match_id

    for i in range(k):
        params = {"less_than_match_id": last_id}
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


def enrich(matches, id_to_name=None):
    total   = len(matches)
    valid   = []
    skipped = 0

    for i, m in enumerate(matches):
        match_id = m["match_id"]
        try:
            detail     = get(f"https://api.opendota.com/api/matches/{match_id}")
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
            valid.append(m)

        except Exception as e:
            print(f"  Failed {match_id}: {e}")
            skipped += 1

        print(f"  Enriched {i+1}/{total} (match {match_id})")
        time.sleep(RATE_LIMIT_DELAY)

    print(f"\nEnrichment: {len(valid)} valid, {skipped} skipped.")
    return valid


def save(matches, json_path, csv_path):
    # sort descending by match_id (newest first) before saving
    matches = sorted(matches, key=lambda x: x["match_id"], reverse=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(matches, f, indent=2)
    print(f"Saved JSON → {json_path} ({len(matches)} matches)")

    csv_fieldnames = [
        "match_id", "patch", "league_name",
        "radiant_name", "dire_name", "radiant_win",
        "draft_order", "draft_teams", "draft_is_pick",
        "radiant_players", "dire_players",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames, extrasaction="ignore")
        writer.writeheader()
        for m in matches:
            pb = m.get("picks_bans", [])
            writer.writerow({
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
            })

    print(f"Saved CSV  → {csv_path}")


if __name__ == "__main__":
    try:
        with open(BASE_DIR / 'data' / 'id_to_name.pikl', 'rb') as file:
            id_to_name = pickle.load(file)
    except Exception:
        print("Error: Need to call collector_hero.py first to update hero information. Exiting...")
        exit(1)

    existing_matches, seen_ids, max_match_id, min_match_id = load_existing(OUTPUT_JSON)

    new_matches = []

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