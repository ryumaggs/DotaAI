import json
import csv

INPUT_JSON = "pro_matches_draft.json"
OUTPUT_CSV = "pro_matches_draft.csv"

def clean(val):
    """Replace None with empty string."""
    if val is None:
        return ""
    return val

def list_to_str(lst):
    """Convert a list to comma-separated string, handling None entries."""
    if not lst:
        return ""
    return ", ".join(str(clean(x)) for x in lst)

with open(INPUT_JSON, encoding="utf-8") as f:
    matches = json.load(f)


if __name__ == "__main__":
    fieldnames = [
        "match_id",
        "league_name",
        "radiant_name",
        "dire_name",
        "radiant_win",
        "radiant_picks",
        "dire_picks",
        "radiant_bans",
        "dire_bans",
        "radiant_players",
        "dire_players",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for m in matches:
            writer.writerow({
                "match_id":        clean(m.get("match_id")),
                "league_name":     clean(m.get("league_name")),
                "radiant_name":    clean(m.get("radiant_name")),
                "dire_name":       clean(m.get("dire_name")),
                "radiant_win":     clean(m.get("radiant_win")),
                "radiant_picks":   list_to_str(m.get("radiant_picks")),
                "dire_picks":      list_to_str(m.get("dire_picks")),
                "radiant_bans":    list_to_str(m.get("radiant_bans")),
                "dire_bans":       list_to_str(m.get("dire_bans")),
                "radiant_players": list_to_str(m.get("radiant_players")),
                "dire_players":    list_to_str(m.get("dire_players")),
            })

    print(f"Saved {len(matches)} rows to {OUTPUT_CSV}")