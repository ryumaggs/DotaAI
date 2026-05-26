import pandas as pd

def parse_list(s):
    """Parse a comma-separated string into a list, stripping whitespace."""
    if pd.isna(s) or s == "":
        return []
    return [x.strip() for x in s.split(",")]