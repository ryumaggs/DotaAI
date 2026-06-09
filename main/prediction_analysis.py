import sys
import pickle
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
import torch
import pandas as pd
from collections import defaultdict
import ast
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
import joblib    
from train import create_model, load_model

def top_k_first_actions(csv_path: str, k: int = 10):
    """
    Prints the top K most likely heroes to be first picked and first banned.
    """


    first_pick_counts = defaultdict(int)
    first_ban_counts  = defaultdict(int)
    total_matches     = 0

    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        try:
            heroes   = [h.strip() for h in row['draft_order'].split(',')]
            is_picks = [s.strip() == 'True' for s in row['draft_is_pick'].split(',')]
        except Exception:
            continue

        total_matches += 1

        for hero, is_pick in zip(heroes, is_picks):
            if is_pick:
                first_pick_counts[hero] += 1
                break

        for hero, is_pick in zip(heroes, is_picks):
            if not is_pick:
                first_ban_counts[hero] += 1
                break

    top_picks = sorted(first_pick_counts.items(), key=lambda x: x[1], reverse=True)[:k]
    top_bans  = sorted(first_ban_counts.items(),  key=lambda x: x[1], reverse=True)[:k]

    print(f"Top {k} first picked heroes (out of {total_matches} matches):")
    for i, (hero, count) in enumerate(top_picks, 1):
        print(f"  {i:2}. {hero:<30} {count:4} times  ({100*count/total_matches:.1f}%)")

    print(f"\nTop {k} first banned heroes (out of {total_matches} matches):")
    for i, (hero, count) in enumerate(top_bans, 1):
        print(f"  {i:2}. {hero:<30} {count:4} times  ({100*count/total_matches:.1f}%)")

def validate_heroes_in_dict(csv_path: str, d: dict):
    import pandas as pd

    df = pd.read_csv(csv_path)
    missing = set()

    for _, row in df.iterrows():
        heroes = [h.strip() for h in row['draft_order'].split(',')]
        for hero in heroes:
            if hero not in d:
                missing.add(hero)

    if missing:
        print(f"Missing {len(missing)} heroes from dictionary:")
        for hero in sorted(missing):
            print(f"  '{hero}'")
    else:
        print("All heroes present in dictionary.")

def print_top_k_predictions(pred, id_to_name_dict, probs):
    for i, pred_id in enumerate(pred):
        print(i+1, id_to_name_dict[pred_id], probs[pred_id])

def print_model_output():
    with open(BASE_DIR / "data" / "name_id_index_maps" / "name_to_index.pikl", 'rb') as file:
        name_to_id_dict = pickle.load(file)
    with open(BASE_DIR / "data" / "name_id_index_maps" / "index_to_name.pikl", 'rb') as file:
        id_to_name_dict = pickle.load(file)  
    top_k_first_actions(csv_path=BASE_DIR / "data" / "pro_matches_draft_objectives.csv")
    save_path =  BASE_DIR / "saved_models" / "draft_transformer.pt"
    device = torch.device('cuda:0')
    num_heroes = len(name_to_id_dict)+1
    base_model = create_model(num_heroes)
    load_model(base_model, save_path)
    base_model.to(device)
    beginning_input = torch.tensor([[127]]).to(device)
    base_model.eval()
    with torch.no_grad():
        out = base_model.forward(beginning_input).cpu().squeeze()
        out[127] = float('-inf')
        probs = torch.nn.functional.softmax(out,dim=0)
        print_top_k_predictions(torch.topk(probs,10).indices.numpy(),id_to_name_dict, probs.numpy())
    
def compute_hero_advantages(cur_row, counter_matrix, synergy_matrix, hero_to_idx):
    # Separate picks by team, ignore bans
    team0_picks = []
    team1_picks = []
    draft_order   = parse_list(cur_row["draft_order"])
    draft_teams   = parse_list(cur_row["draft_teams"])
    draft_is_pick = parse_list(cur_row["draft_is_pick"])
    
    for hero, team, is_pick_str in zip(draft_order, draft_teams, draft_is_pick):
        team = int(team)
        is_pick = is_pick_str.strip().lower() == "true"
        if not is_pick:
            continue
        if team == 0:
            team0_picks.append(hero)
        else:
            team1_picks.append(hero)
    
    def hero_advantage(hero, teammates, enemies):
        idx = hero_to_idx[hero]
        
        synergy = sum(
            synergy_matrix[idx][hero_to_idx[mate]]
            for mate in teammates if mate != hero
        )
        counter = sum(
            counter_matrix[idx][hero_to_idx[enemy]]
            for enemy in enemies
        )
        return synergy/2 + counter
    
    team0_advantages = {}
    team1_advantages = {}
    for hero in team0_picks:
        team0_advantages[hero] = hero_advantage(hero, team0_picks, team1_picks)
    for hero in team1_picks:
        team1_advantages[hero] = hero_advantage(hero, team1_picks, team0_picks)
    
    return team0_advantages, team1_advantages

def compute_confusion_matrix(counter_matrix, synergy_matrix, name_to_index_dict):
    predicted_wins = {} #first is actual, second is predicted
    predicted_wins[(0,0)] = 0
    predicted_wins[(0,1)] = 0
    predicted_wins[(1,0)] = 0
    predicted_wins[(1,1)] = 0

    advantage_scores = {}
    for k in predicted_wins:
        advantage_scores[k] = []

    for cur_index in tqdm(range(len(df))):
        cur_row = df.iloc[cur_index]
        
        team0_adv, team1_adv = compute_hero_advantages(cur_row, counter_matrix, synergy_matrix, name_to_index_dict)
        team0sum = sum((list(team0_adv.values())))
        team1sum = sum(list(team1_adv.values()))
        if cur_row['radiant_win']:
            if team0sum > team1sum: #correct radiant win
                predicted_wins[(0,0)] += 1
                advantage_scores[(0,0)].append((team0sum - team1sum).item())
            else:
                predicted_wins[(0,1)] += 1
                advantage_scores[(0,1)].append((team0sum - team1sum).item())
        elif not cur_row['radiant_win']:
            if team0sum > team1sum: #correct radiant win
                predicted_wins[(1,0)] += 1
                advantage_scores[(1,0)].append((team0sum - team1sum).item())
            else:
                predicted_wins[(1,1)] += 1
                advantage_scores[(1,1)].append((team0sum - team1sum).item())
    
    for k in advantage_scores:
        print(k, np.mean(advantage_scores[k]), np.std(advantage_scores[k]))

    plot_histogram(advantage_scores, 100, "./test_hist.png")
 
def plot_histogram(data: dict[str, list[float]], k: int, save_path: str) -> None:
    """
    Plot a histogram with each key in `data` as a distinct color.
 
    Args:
        data:      Dict with exactly 4 keys; each value is a list of floats.
        k:         Number of bins.
        save_path: File path where the histogram image will be saved.
    """
    if len(data) != 4:
        raise ValueError(f"Expected exactly 4 keys, got {len(data)}.")
 
    colors = ["#4C5BB0", "#F10B0B", "#02A027", "#FFEE00"]  # blue, orange, green, red
 
    fig, ax = plt.subplots(figsize=(9, 5))
 
    for (label, values), color in zip(data.items(), colors):
        ax.hist(values, bins=k, color=color, alpha=0.65, edgecolor="white",
                linewidth=0.6, label=label)
 
    ax.set_xlabel("Value", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Histogram by Group", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(title="Group", fontsize=10, title_fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
 
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved histogram to: {save_path}")

def train_calibrator(counter_matrix, synergy_matrix, name_to_index_dict):
    test_stop_index = int(0.15*len(df))

    test_scores, test_labels = [], []
    all_scores, all_labels = [], []
    for cur_index in tqdm(range(len(df))):
        
        cur_row = df.iloc[cur_index]
        team0_adv, team1_adv = compute_hero_advantages(cur_row, counter_matrix, synergy_matrix, name_to_index_dict)
        team0sum = sum((list(team0_adv.values())))
        team1sum = sum(list(team1_adv.values()))
        advantage_score = team0sum - team1sum
        if cur_index < test_stop_index:
            test_scores.append(advantage_score)
            test_labels.append(int(cur_row['radiant_win']))
        else:
            all_scores.append(advantage_score)
            all_labels.append(int(cur_row['radiant_win']))

    all_scores = np.array(all_scores).reshape(-1, 1)
    all_labels = np.array(all_labels)

    cal = LogisticRegression(verbose=1)
    cal.fit(all_scores, all_labels)

    # Save
    joblib.dump(cal, "calibrator.pkl")

    return cal, np.array(test_scores).reshape(-1, 1), np.array(test_labels) #all indices before this are considered testing and were not seen in training
          
if __name__ == "__main__":
    from main.util import parse_list
    df = pd.read_csv(BASE_DIR / "data" / "pro_matches_draft_objectives.csv")
    counter_matrix = torch.load(BASE_DIR / 'data' / 'counter_matrix.pt')
    synergy_matrix = torch.load(BASE_DIR / 'data' / 'synergy_matrix.pt')
    name_to_index_path = BASE_DIR / 'data' / "name_id_index_maps" / "name_to_index.pikl"
    with open(name_to_index_path, 'rb') as file:
        name_to_index_dict = pickle.load(file)

    import joblib

    cal, X_test, y_test = train_calibrator(counter_matrix, synergy_matrix, name_to_index_dict)

    prob_pred = cal.predict_proba(X_test)[:, 1]  # shape (N,)

    fraction_pos, mean_pred = calibration_curve(y_test, prob_pred, n_bins=10)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mean_pred, fraction_pos, marker="o", label="Calibrator")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction Positive")
    ax.set_title("Calibration Curve")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig("calibration_curve.png", dpi=150)
        
    from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

    y_pred = (prob_pred >= 0.5).astype(int)
    print("Accuracy:   ", accuracy_score(y_test, y_pred))
    print("Brier Score:", brier_score_loss(y_test, prob_pred))  # lower is better, 0=perfect
    print("Log Loss:   ", log_loss(y_test, prob_pred))   

    



