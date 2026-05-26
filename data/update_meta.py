import json
import os
META_FILE = "pro_matches_meta.json"

def init_meta_from_existing(json_path, META_FILE):
    with open(json_path, encoding="utf-8") as f:
        matches = json.load(f)

    if not matches:
        print("No matches found in JSON.")
        return

    match_ids = [m["match_id"] for m in matches]

    meta = {
        "max_match_id": max(match_ids),
        "min_match_id": min(match_ids),
        "gaps": []
    }

    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Meta initialized: max={meta['max_match_id']}, min={meta['min_match_id']}")

def load_meta():
    if not os.path.exists(META_FILE):
        return {"max_match_id": None, "min_match_id": None, "gaps": []}
    with open(META_FILE, encoding="utf-8") as f:
        return json.load(f)


def save_meta(meta):
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def update_meta(meta, new_matches):
    if not new_matches:
        return meta

    new_ids       = [m["match_id"] for m in new_matches]
    new_max       = max(new_ids)
    new_min       = min(new_ids)
    old_max       = meta["max_match_id"]
    old_min       = meta["min_match_id"]

    # first run
    if old_max is None:
        meta["max_match_id"] = new_max
        meta["min_match_id"] = new_min
        return meta

    # if fetching newer games and a gap exists between old_max and new_min
    if new_max > old_max and new_min > old_max:
        # full gap — none of the middle ground was collected
        meta["gaps"].append({"upper": new_min, "lower": old_max})
        meta["max_match_id"] = new_max

    elif new_max > old_max:
        # no gap — overlap was found, clean connection
        meta["max_match_id"] = new_max

    # if fetching older games, just update min
    if new_min < old_min:
        meta["min_match_id"] = new_min

    return meta

if __name__ == "__main__":
    init_meta_from_existing("pro_matches_draft.json", META_FILE)