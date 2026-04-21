import pandas as pd
import re

# ---- 1. Define category mapping (STANDARD NAMES) ----
CATEGORY_MAP = {
    "computers": "computers",
    "laptops": "computers",
    "pc laptops": "computers",
    "2 in 1 laptops": "computers",
    "computer components": "computers",

    "hard drives": "storage",
    "internal hard drives": "storage",
    "solid state drives": "storage",
    "external drives": "storage",
    "data storage": "storage",
    "nas servers": "storage",
    "network attached storage drives": "storage",

    "tv": "tv_video",
    "television": "tv_video",
    "tv & video": "tv_video",
    "home theater": "tv_video",
    "blu-ray players": "tv_video",
    "dvd players": "tv_video",

    "cell phones": "mobile",
    "cell phone accessories": "mobile",
    "chargers": "mobile",
    "power accessories": "mobile",

    "camera": "camera",
    "camera lenses": "camera",
    "photo accessories": "camera",

    "audio": "audio",
    "home audio": "audio",
    "speakers": "audio",
}

# ---- 2. Cleaning function ----
def clean_and_map_categories(cat_string):
    if pd.isna(cat_string):
        return []

    categories = cat_string.split(",")

    cleaned = []

    for cat in categories:
        # Normalize
        cat = cat.strip().lower()

        # Remove garbage characters
        cat = re.sub(r"[^a-z0-9\s&/-]", "", cat)

        # Remove short/noisy tokens
        if len(cat) < 3:
            continue
        if not re.search(r"[aeiou]", cat):
            continue

        # ---- 3. Map to standard category ----
        mapped = None
        for key in CATEGORY_MAP:
            if key in cat:
                mapped = CATEGORY_MAP[key]
                break

        if mapped:
            cleaned.append(mapped)
        else:
            cleaned.append(cat)  # keep cleaned version if no mapping

    # Remove duplicates
    cleaned = list(dict.fromkeys(cleaned))

    return cleaned

