import gzip
import json
import urllib.request
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parent
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "data" / "processed"
REVIEWS_PATH = RAW_DIR / "Digital_Music_5.json.gz"

REVIEWS_URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/"
    "categoryFilesSmall/Digital_Music_5.json.gz"
)


RAW_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not REVIEWS_PATH.exists():
    urllib.request.urlretrieve(REVIEWS_URL, REVIEWS_PATH)

review_rows = []
with gzip.open(REVIEWS_PATH, "rt", encoding="utf-8") as file:
    for line in file:
        review = json.loads(line)
        style = review.get("style") or {}
        item_format = style.get("Format:", "unknown")
        review_rows.append(
            {
                "user_id": review["reviewerID"],
                "item_id": review["asin"],
                "rating": float(review["overall"]),
                "timestamp": int(review["unixReviewTime"]),
                "item_format": str(item_format).strip() or "unknown",
            }
        )

events = pd.DataFrame(review_rows)
raw_interaction_count = len(events)
raw_user_count = events["user_id"].nunique()
raw_item_count = events["item_id"].nunique()
raw_unknown_format_count = int((events["item_format"] == "unknown").sum())
duplicate_interactions_removed = int(events.duplicated(["user_id", "item_id"]).sum())

events = events.sort_values("timestamp")
events = events.drop_duplicates(["user_id", "item_id"], keep="last")
events = events.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

after_dedup_interaction_count = len(events)
after_dedup_user_count = events["user_id"].nunique()
after_dedup_item_count = events["item_id"].nunique()

events["user_count"] = events.groupby("user_id")["item_id"].transform("size")
events["position"] = events.groupby("user_id").cumcount() + 1
events["split"] = "train"
events.loc[
    (events["user_count"] >= 3)
    & (events["position"] == events["user_count"] - 1),
    "split",
] = "validation"
events.loc[
    (events["user_count"] >= 2)
    & (events["position"] == events["user_count"]),
    "split",
] = "test"

train_positive = events[
    (events["split"] == "train") & (events["rating"] >= 4)
]
candidate_items = train_positive["item_id"].value_counts().head(5000).index
events = events[events["item_id"].isin(candidate_items)].copy()
after_candidate_interaction_count = len(events)
after_candidate_user_count = events["user_id"].nunique()
after_candidate_item_count = events["item_id"].nunique()

known_users = set(events.loc[events["split"] == "train", "user_id"])
events = events[events["user_id"].isin(known_users)].copy()

train_last_time = events.loc[events["split"] == "train"].groupby("user_id")[
    "timestamp"
].max()
events["train_last_time"] = events["user_id"].map(train_last_time)
events["context_month"] = pd.to_datetime(
    events["train_last_time"], unit="s"
).dt.month
train_rows = events["split"] == "train"
events.loc[train_rows, "context_month"] = pd.to_datetime(
    events.loc[train_rows, "timestamp"], unit="s"
).dt.month

train_events = events[events["split"] == "train"]
items = train_events[train_events["item_format"] != "unknown"].groupby(
    ["item_id", "item_format"]
).size().reset_index(name="count")
items = items.sort_values("count", ascending=False).drop_duplicates("item_id")
items = items.set_index("item_id").reindex(candidate_items)
items.index.name = "item_id"
items["item_format"] = items["item_format"].fillna("unknown")
items = items[["item_format"]].reset_index()
events = events[
    ["user_id", "item_id", "rating", "timestamp", "split", "context_month"]
]
final_interaction_count = len(events)
final_user_count = events["user_id"].nunique()
final_item_count = events["item_id"].nunique()

preparation_summary = pd.DataFrame(
    {
        "Этап": [
            "Исходные отзывы",
            "После удаления дублей",
            "После выбора 5 000 объектов",
            "После фильтрации пользователей без train",
        ],
        "Взаимодействия": [
            raw_interaction_count,
            after_dedup_interaction_count,
            after_candidate_interaction_count,
            final_interaction_count,
        ],
        "Пользователи": [
            raw_user_count,
            after_dedup_user_count,
            after_candidate_user_count,
            final_user_count,
        ],
        "Объекты": [
            raw_item_count,
            after_dedup_item_count,
            after_candidate_item_count,
            final_item_count,
        ],
    }
)

events.to_csv(OUTPUT_DIR / "events.csv", index=False)
items.to_csv(OUTPUT_DIR / "items.csv", index=False)
