from torch.utils.data import DataLoader, TensorDataset, random_split
import pickle
import pandas as pd
import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from main.util import parse_list
import torch


def df_to_tensors_full_draft(df, name_to_id):
    """
    Convert a dataframe of matches into tensors ready for the model.

    Args:
        df:         pandas DataFrame with columns as described
        name_to_id: dict mapping hero name -> OpenDota hero ID

    Returns:
        hero_ids:  (N, 24) int tensor  — hero ID per draft slot
        team_ids:  (N, 24) int tensor  — 0=radiant, 1=dire per slot
        is_picks:  (N, 24) bool tensor — True if pick, False if ban
        labels:    (N,)    float tensor — 1.0 if radiant win
    """
    all_hero_ids = []
    all_team_ids = []
    all_is_picks = []
    all_labels   = []
    skipped      = 0

    for _, row in df.iterrows():
        draft_order   = parse_list(row["draft_order"])
        draft_teams   = parse_list(row["draft_teams"])
        draft_is_pick = parse_list(row["draft_is_pick"])

        # skip malformed rows
        if not (len(draft_order) == len(draft_teams) == len(draft_is_pick)):
            skipped += 1
            continue

        # truncate or pad to exactly 24 tokens
        seq_len = 24
        hero_ids = []
        team_ids = []
        is_picks = []

        for i in range(seq_len):
            if i < len(draft_order):
                hero_name = draft_order[i]
                hero_id   = name_to_id.get(hero_name, 0)  # 0 = PAD/unknown
                team      = int(draft_teams[i])
                is_pick   = draft_is_pick[i].strip().lower() == "true"
            else:
                # pad if sequence shorter than 24
                hero_id = 0
                team    = 0
                is_pick = False

            hero_ids.append(hero_id)
            team_ids.append(team)
            is_picks.append(is_pick)

        all_hero_ids.append(hero_ids)
        all_team_ids.append(team_ids)
        all_is_picks.append(is_picks)

        label = 1.0 if str(row["radiant_win"]).strip().lower() == "true" else 0.0
        all_labels.append(label)

    if skipped:
        print(f"Warning: skipped {skipped} malformed rows.")

    hero_ids_t = torch.tensor(all_hero_ids, dtype=torch.long)   # (N, 24)
    team_ids_t = torch.tensor(all_team_ids, dtype=torch.long)   # (N, 24)
    is_picks_t = torch.tensor(all_is_picks, dtype=torch.bool)   # (N, 24)
    labels_t   = torch.tensor(all_labels,   dtype=torch.float)  # (N,)

    return hero_ids_t, team_ids_t, is_picks_t, labels_t

def df_to_tensors_picks_only(df, name_to_id):
    """
    Converts dataframe to tensors containing only the 10 picked heroes,
    in the order they were selected.

    Returns:
        hero_ids:  (N, 10) int tensor  — hero ID per pick slot in selection order
        team_ids:  (N, 10) int tensor  — [0,0,0,0,0,1,1,1,1,1]
        is_picks:  (N, 10) bool tensor — all True
        labels:    (N,)    float tensor
    """
    TEAM_IDS = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    IS_PICKS  = [True] * 10

    all_hero_ids = []
    all_labels   = []
    skipped      = 0

    for _, row in df.iterrows():
        draft_order   = parse_list(row["draft_order"])
        draft_teams   = parse_list(row["draft_teams"])
        draft_is_pick = parse_list(row["draft_is_pick"])

        if not (len(draft_order) == len(draft_teams) == len(draft_is_pick)):
            skipped += 1
            continue

        # extract picks only, preserving selection order, split by team
        radiant_picks = []
        dire_picks    = []

        for hero_name, team, is_pick in zip(draft_order, draft_teams, draft_is_pick):
            if is_pick.strip().lower() != "true":
                continue
            hero_id = name_to_id.get(hero_name, 0)
            if int(team) == 0:
                radiant_picks.append(hero_id)
            else:
                dire_picks.append(hero_id)

        # pad to 5 each if somehow incomplete
        while len(radiant_picks) < 5:
            radiant_picks.append(0)
        while len(dire_picks) < 5:
            dire_picks.append(0)

        all_hero_ids.append(radiant_picks[:5] + dire_picks[:5])

        label = 0.0 if str(row["radiant_win"]).strip().lower() == "true" else 1.0
        all_labels.append(label)

    if skipped:
        print(f"Warning: skipped {skipped} malformed rows.")

    N = len(all_hero_ids)

    hero_ids_t = torch.tensor(all_hero_ids,          dtype=torch.long)
    team_ids_t = torch.tensor([TEAM_IDS] * N,        dtype=torch.long)
    is_picks_t = torch.tensor([IS_PICKS]  * N,       dtype=torch.bool)
    labels_t   = torch.tensor(all_labels,            dtype=torch.float)

    return hero_ids_t, team_ids_t, is_picks_t, labels_t

def augment_dataset(dataset):
    hero_ids, team_ids, is_picks, labels = dataset.tensors
    
    # flip team ids (0->1, 1->0) and invert labels
    aug_team_ids = 1 - team_ids
    aug_labels = 1 - labels
    
    # concatenate original and augmented
    new_hero_ids = torch.cat([hero_ids, hero_ids], dim=0)
    new_team_ids = torch.cat([team_ids, aug_team_ids], dim=0)
    new_is_picks = torch.cat([is_picks, is_picks], dim=0)
    new_labels   = torch.cat([labels, aug_labels], dim=0)
    
    return TensorDataset(new_hero_ids, new_team_ids, new_is_picks, new_labels)

def load_data(path_to_csv,
              name_to_id_path):
    df = pd.read_csv(path_to_csv)
    with open(name_to_id_path, 'rb') as file:
        name_to_id_dict = pickle.load(file)
    
    hero_ids_t, team_ids_t, is_picks_t, labels_t = df_to_tensors_full_draft(df, name_to_id_dict)
    dataset = TensorDataset(hero_ids_t, team_ids_t, is_picks_t, labels_t)

    dataset = augment_dataset(dataset)

    return dataset

def convert_to_torch_dataset(dataset):
    # define split sizes
    total     = len(dataset)
    train_len = int(0.7 * total)
    val_len   = int(0.15 * total)
    test_len  = total - train_len - val_len

    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=32, shuffle=False)
    test_loader  = DataLoader(test_set,  batch_size=32, shuffle=False)

    return train_loader, val_loader, test_loader

class DraftDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_csv, name_to_id_path):
        dataset = load_data(path_to_csv, name_to_id_path)
        self.hero_ids = dataset.tensors[0]  # (N, 24)

    def __len__(self):
        return self.hero_ids.size(0)

    def __getitem__(self, idx):
        sequence = self.hero_ids[idx]  # (24,)
        x = sequence[:-1]              # (23,)
        y = sequence[1:]               # (23,)
        return x, y