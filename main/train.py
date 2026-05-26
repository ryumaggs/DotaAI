import sys
import pickle
from torch.utils.data import DataLoader, TensorDataset, random_split
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import pandas as pd
from main.transformer import *
from main.util import parse_list

DEVICE = 'cuda:0'
def df_to_tensors(df, name_to_id):
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

def load_data(path_to_csv,
              name_to_id_path):
    df = pd.read_csv(path_to_csv)
    with open(name_to_id_path, 'rb') as file:
        name_to_id_dict = pickle.load(file)
    
    hero_ids_t, _, _, labels_t = df_to_tensors(df, name_to_id_dict)
    dataset = TensorDataset(hero_ids_t, labels_t)
    dataset = TensorDataset(hero_ids_t, labels_t)

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

def create_model():
    model = DraftTransformer(num_heroes=256, 
                 embed_dim=64, 
                 num_heads=4, 
                 ff_dim=128, 
                 num_layers=2, dropout=0.1)
    criterion = nn.BCEWithLogitsLoss()
    return model, criterion

def load_model(base_model):
    pass

def save_model():
    pass

def new_training(epochs):
    train_loader, val_loader, test_loader = load_data(BASE_DIR / "data" / "pro_matches_draft.csv",
                       BASE_DIR / "data" / "name_to_id.pikl")
    base_model, criterion = create_model()
    device = torch.device(DEVICE)
    if DEVICE != 'cpu':
        base_model.to(device)

    base_optimizer = torch.optim.AdamW(base_model.parameters(), lr=1e-4, weight_decay=0.01)

    for e in range(epochs):
        for hero_ids, labels in train_loader:
            base_model.train()
            preds = base_model(hero_ids.to(device))
            loss  = criterion(preds, labels)
            loss.backward()
            base_optimizer.step()
        
        for hero_ids, labels in val_loader:
            base_model.eval()
            with torch.no_grad():
                preds = base_model(hero_ids.to(device))
                loss  = criterion(preds, labels)
                print('val loss: ', loss.item())

        torch.save(base_model.state_dict(), BASE_DIR / "saved_models" / "draft_transformer.pt")
        
def continue_training():
    base_model = create_model()
    load_model(base_model)
    new_training()
    pass

if __name__ == "__main__":
    new_training(epochs=5)