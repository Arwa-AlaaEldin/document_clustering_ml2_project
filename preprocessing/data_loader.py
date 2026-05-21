
import os
import re
import pandas as pd


CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)


# text cleaning
# remove emails, URLs, numbers, excess whitespace
def clean_text(text):

    # emails
    text = re.sub(r"\S+@\S+", " ", text)
    # URLs            
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    # keep only letters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    # drop 1-2 char words
    text = re.sub(r"\b[a-zA-Z]{1,2}\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def clean_batch(texts: list[str]):
    return [clean_text(t) for t in texts]


# 20 Newsgroups
# load 20 Newsgroups (uses cache if available)
# returns DataFrame with columns: text, clean_text, label, category
def load_newsgroups(subset="all"):
    
    cache_path = os.path.join(CACHE_DIR, f"newsgroups_{subset}.parquet")

    if os.path.exists(cache_path):
        print(f"[Data] Loading newsgroups from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    print("[Data] Fetching 20 Newsgroups from sklearn (first time only)…")
    from sklearn.datasets import fetch_20newsgroups
    raw = fetch_20newsgroups(subset=subset, remove=("headers", "footers", "quotes"))

    df = pd.DataFrame({
        "text": raw.data,
        "label": raw.target.astype(int),
        "category": [raw.target_names[t] for t in raw.target],
    })
    df["clean_text"] = clean_batch(df["text"].tolist())

    # drop very short docs
    df = df[df["clean_text"].str.split().str.len() >= 20].reset_index(drop=True)

    df.to_parquet(cache_path, index=False)
    print(f"[Data] Saved to cache. {len(df):,} docs, {df['label'].nunique()} categories.")
    return df


# Wikipedia People
# load People Wikipedia (uses cache if available)
# returns DataFrame with columns: name, text, clean_text
def load_wikipedia(path):
  
    cache_path = os.path.join(CACHE_DIR, "wikipedia.parquet")

    if os.path.exists(cache_path):
        print(f"[Data] Loading wikipedia from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"[Data] Reading Wikipedia from {path}…")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        df = pd.read_json(path, lines=True)
    elif ext == ".json":
        df = pd.read_json(path)
    else:
        df = pd.read_csv(path)

    # normalise column names
    col_map = {}
    for col in df.columns:
        low = col.lower()
        if low in ("text", "content", "article", "bio", "biography", "abstract"):
            col_map[col] = "text"
        elif low in ("name", "person", "title", "subject"):
            col_map[col] = "name"
    df = df.rename(columns=col_map)

    df["text"] = df["text"].fillna("").astype(str)
    df["name"] = df.get("name", pd.Series("", index=df.index)).fillna("").astype(str)
    df["clean_text"] = clean_batch(df["text"].tolist())

    df = df[df["clean_text"].str.split().str.len() >= 20].reset_index(drop=True)

    df.to_parquet(cache_path, index=False)
    print(f"[Data] Saved to cache. {len(df):,} docs.")
    return df