import pandas as pd
import torch

def compute_matchup_matrices(df, name_to_index):
        """
        Returns two tensors of shape [N, N]:
        
        counter_matrix[i, j] = how much better hero i performs against hero j
                            relative to hero i's average performance vs all opponents.
                            0.0 where unseen.
        
        synergy_matrix[i, j] = how much better hero i performs alongside hero j
                            relative to hero i's average performance alongside anyone.
                            0.0 where unseen.
        """
        N = len(name_to_index)

        # raw accumulators for counter (opposite team)
        counter_wins   = torch.zeros(N, N)
        counter_totals = torch.zeros(N, N)

        # raw accumulators for synergy (same team)
        synergy_wins   = torch.zeros(N, N)
        synergy_totals = torch.zeros(N, N)

        for _, row in df.iterrows():
            try:
                heroes      = [h.strip() for h in row['draft_order'].split(',')]
                teams       = [int(t.strip()) for t in row['draft_teams'].split(',')]
                is_picks    = [s.strip() == 'True' for s in row['draft_is_pick'].split(',')]
                radiant_win = str(row['radiant_win']).strip() == 'True'
            except Exception:
                continue

            picked = [
                (hero, team)
                for hero, team, is_pick in zip(heroes, teams, is_picks)
                if is_pick and hero in name_to_index
            ]

            for idx_a, (hero_a, team_a) in enumerate(picked):
                for idx_b, (hero_b, team_b) in enumerate(picked):
                    if idx_a == idx_b:
                        continue

                    i = name_to_index[hero_a]
                    j = name_to_index[hero_b]
                    hero_a_won = (team_a == 0 and radiant_win) or \
                                (team_a == 1 and not radiant_win)

                    if team_a != team_b:
                        # counter: opposite teams
                        counter_totals[i, j] += 1
                        if hero_a_won:
                            counter_wins[i, j] += 1
                    else:
                        # synergy: same team
                        synergy_totals[i, j] += 1
                        if hero_a_won:
                            synergy_wins[i, j] += 1

        # --- raw winrates ---
        counter_seen = counter_totals > 0
        synergy_seen = synergy_totals > 0

        counter_wr = torch.zeros(N, N)
        synergy_wr = torch.zeros(N, N)

        counter_wr[counter_seen] = counter_wins[counter_seen] / counter_totals[counter_seen]
        synergy_wr[synergy_seen] = synergy_wins[synergy_seen] / synergy_totals[synergy_seen]

        # --- baseline winrates per hero ---
        # counter baseline: hero i's average winrate across all opponents
        counter_baseline = torch.zeros(N)
        for i in range(N):
            seen_cols = counter_seen[i]
            if seen_cols.any():
                counter_baseline[i] = counter_wr[i, seen_cols].mean()

        # synergy baseline: hero i's average winrate alongside anyone
        synergy_baseline = torch.zeros(N)
        for i in range(N):
            seen_cols = synergy_seen[i]
            if seen_cols.any():
                synergy_baseline[i] = synergy_wr[i, seen_cols].mean()

        # --- advantage = winrate - baseline ---
        # unseen pairs stay at 0.0 (neutral, no information)
        counter_matrix = torch.zeros(N, N)
        synergy_matrix = torch.zeros(N, N)

        counter_matrix[counter_seen] = (
            counter_wr[counter_seen] - counter_baseline.unsqueeze(1).expand(N, N)[counter_seen]
        )
        synergy_matrix[synergy_seen] = (
            synergy_wr[synergy_seen] - synergy_baseline.unsqueeze(1).expand(N, N)[synergy_seen]
        )

        return counter_matrix, synergy_matrix  # each [N, N]

if __name__ == "__main__":
    import pickle
    path_to_csv = "./pro_matches_draft_objectives.csv"
    name_to_index_path = "./name_id_index_maps/name_to_index.pikl"
    data_df = pd.read_csv(path_to_csv)
    with open(name_to_index_path, 'rb') as file:
        name_to_index_dict = pickle.load(file)

    counter_matrix, synergy_matrix = compute_matchup_matrices(data_df, name_to_index_dict)

    torch.save(counter_matrix, 'counter_matrix.pt')
    torch.save(synergy_matrix, 'synergy_matrix.pt')