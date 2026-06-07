import pandas as pd
import ast
import pickle
import torch
from collections import defaultdict

def parse_draft(row):
    """Parse draft fields into a list of (hero, team, is_pick) tuples."""
    heroes = [h.strip() for h in row['draft_order'].split(',')]
    teams = [int(t.strip()) for t in row['draft_teams'].split(',')]
    is_picks = [s.strip() == 'True' for s in row['draft_is_pick'].split(',')]
    return list(zip(heroes, teams, is_picks))

def compute_hero_stats(df):
    """
    Returns a dict: hero -> {
        pick_count, ban_count, appearance_count,
        pick_slots, ban_slots,
        win_count, loss_count
    }
    """
    stats = defaultdict(lambda: {
        'pick_count': 0,
        'ban_count': 0,
        'pick_slots': [],   # draft slot indices when picked
        'ban_slots': [],    # draft slot indices when banned
        'win_count': 0,
        'loss_count': 0,
    })

    for _, row in df.iterrows():
        try:
            draft = parse_draft(row)
            radiant_win = str(row['radiant_win']).strip() == 'True'
        except Exception:
            continue

        for slot_idx, (hero, team, is_pick) in enumerate(draft):
            s = stats[hero]
            # team: 0 = radiant, 1 = dire
            hero_won = (team == 0 and radiant_win) or (team == 1 and not radiant_win)

            if is_pick:
                s['pick_count'] += 1
                s['pick_slots'].append(slot_idx)
                if hero_won:
                    s['win_count'] += 1
                else:
                    s['loss_count'] += 1
            else:
                s['ban_count'] += 1
                s['ban_slots'].append(slot_idx)

    return stats

def compute_pick_rate(stats, total_matches):
    """Pick rate = times picked / total matches."""
    return {
        hero: s['pick_count'] / total_matches
        for hero, s in stats.items()
    }

def compute_ban_rate(stats, total_matches):
    """Ban rate = times banned / total matches."""
    return {
        hero: s['ban_count'] / total_matches
        for hero, s in stats.items()
    }


def compute_avg_pick_slot(stats):
    """Average draft slot index (0-23) when the hero is picked."""
    result = {}
    for hero, s in stats.items():
        slots = s['pick_slots']
        if slots:
            result[hero] = sum(slots) / len(slots)
    return result


def compute_avg_ban_slot(stats):
    """Average draft slot index (0-23) when the hero is banned."""
    result = {}
    for hero, s in stats.items():
        slots = s['ban_slots']
        if slots:
            result[hero] = sum(slots) / len(slots)
    return result

def compute_win_rate(stats):
    """Win rate = wins / (wins + losses), only over games where hero was picked."""
    result = {}
    for hero, s in stats.items():
        total = s['win_count'] + s['loss_count']
        if total > 0:
            result[hero] = s['win_count'] / total
    return result

def build_meta_stats(csv_path):
    df = pd.read_csv(csv_path)
    total_matches = len(df)

    stats = compute_hero_stats(df)

    return {
        'pick_rate':      compute_pick_rate(stats, total_matches),
        'ban_rate':       compute_ban_rate(stats, total_matches),
        'avg_pick_slot':  compute_avg_pick_slot(stats),
        'avg_ban_slot':   compute_avg_ban_slot(stats),
        'win_rate':       compute_win_rate(stats),
        'raw':            stats,  # keep raw counts for debugging
    }

def build_meta_tensor(csv_path: str, name_to_id_path: str, id_to_name_path: str, num_slots: int = 23):
    """
    Returns a FloatTensor of shape [N_heroes, 5] where columns are:
        0: pick_rate
        1: ban_rate
        2: avg_pick_slot (normalized 0-1)
        3: avg_ban_slot  (normalized 0-1)
        4: win_rate

    Heroes not seen in the data get zeros.
    
    Args:
        csv_path:         path to pro match CSV
        name_to_id_path:  path to name_to_id .pkl file  {hero_name: hero_id}
        id_to_name_path:  path to id_to_name .pkl file  {hero_id: hero_name}
        num_slots:        total draft slots for normalization (default 23)
    """
    with open(name_to_id_path, 'rb') as f:
        name_to_id = pickle.load(f)
    with open(id_to_name_path, 'rb') as f:
        id_to_name = pickle.load(f)

    print(id_to_name)
    exit(1)

    df = pd.read_csv(csv_path)
    total_matches = len(df)
    stats = compute_hero_stats(df)

    pick_rate = compute_pick_rate(stats, total_matches)
    ban_rate  = compute_ban_rate(stats, total_matches)
    avg_pick  = compute_avg_pick_slot(stats)
    avg_ban   = compute_avg_ban_slot(stats)
    win_rate  = compute_win_rate(stats)

    N = len(name_to_id)
    tensor = torch.zeros(N, 5)

    for hero_id, hero_name in id_to_name.items():
        if hero_name not in name_to_id:
            print(f"Warning: '{hero_name}' in id_to_name but missing from name_to_id, skipping")
            continue

        tensor[hero_id, 0] = pick_rate.get(hero_name, 0.0)
        tensor[hero_id, 1] = ban_rate.get(hero_name, 0.0)
        tensor[hero_id, 2] = avg_pick.get(hero_name, 0.0) / num_slots
        tensor[hero_id, 3] = avg_ban.get(hero_name, 0.0)  / num_slots
        tensor[hero_id, 4] = win_rate.get(hero_name, 0.0)

    return tensor  # [N_heroes, 5]

if __name__ == "__main__":
    #out = build_meta_stats('../data/pro_matches_draft_objectives.csv')
    #print(len(out['win_rate']))
    build_meta_tensor('../data/pro_matches_draft_objectives.csv',
                      name_to_id_path="../data/name_to_id.pikl",
                      id_to_name_path="../data/name_to_id.pikl",
                      num_slots=23,)