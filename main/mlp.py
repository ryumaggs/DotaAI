import torch
import torch.nn as nn

class DraftMLP(nn.Module):
    def __init__(self, n_heroes=200):
        super().__init__()
        # Binary presence vector for each team
        self.n_heroes = n_heroes
        self.net = nn.Sequential(
            nn.Linear(n_heroes * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1)
        )
    
    def process_data(self, all_data_info, device=torch.device('cpu')):
        radiant_picks = None
        dire_picks = None

        hero_ids, team_ids, is_picked, _ = all_data_info
        radiant_picks = torch.zeros(hero_ids.shape[0],self.n_heroes)
        picks = hero_ids[(team_ids == 0) & is_picked].view(hero_ids.size(0), -1)  # (B, 5)
        radiant_picks.scatter_(1, picks.long(), 1)

        dire_picks = torch.zeros(hero_ids.shape[0],self.n_heroes)
        picks = hero_ids[(team_ids == 1) & is_picked].view(hero_ids.size(0), -1)  # (B, 5)
        dire_picks.scatter_(1, picks.long(), 1)


        return radiant_picks.to(device), dire_picks.to(device)
    
    def forward(self, processed_data_output):
        # radiant_picks, dire_picks: (B, n_heroes) binary
        radiant_picks, dire_picks = processed_data_output
        x = torch.cat([radiant_picks, dire_picks], dim=-1)
        return self.net(x).squeeze(1)