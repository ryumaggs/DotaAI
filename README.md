# Dota Draft and Objectives Sequence Predictor

Focuses only on professional matches in the ESPORTS scene of the game Dota2

Predicts Hero Draft (24 hero sequence) with 85%+ top-5 accuracy

Predicts objectives taken sequence of game (WIP) culminating in win or loss

## Model Architecture

### Draft predictor:
- Small decoder transformer
- Vocabulary: number of heros in Dota 2
- Sequence length, 24 heros. (total of 10 heros picked and 14 heroes banned in captains mode alternating sequence)

### Objectives sequence predictor
Objectives in Dota2 refer to Towers, Roshan, Barracks, and the Ancient (win/lose). 

- Decoder transformer with cross attention to draft predictor sequence encoding.
- Vocabulary: (Roshan x3) + (Towers x 2-teams) + (6 barracks x 2 teams) + (2 ancient)
- Sequence length: Varrying depending on how the game proceeds.

## Data

### OpenDota
- Data is collected using REST API via [OpenDota](https://www.opendota.com/).
- You can put your own API key to speed up calls by creating a "secrets" directory in the main dir
  - put your API key as a txt file within the secrets dir
- Can use OpenDota's free no API key calls, but will need to set delay to ~1.1s, longer if you are on VPN 
