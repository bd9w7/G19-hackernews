"""
Hacker News Story Scraper
Data source: Hacker News Firebase API (https://hacker-news.firebaseio.com/v0/)
No API key required. Collects top stories and Ask HN posts.
"""

import requests
import pandas as pd
import json
import time
import re 
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
nltk.download('stopwords')

BASE_URL = "https://hacker-news.firebaseio.com/v0"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "raw_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def fetch_json(url: str, retries: int = 3) -> dict | list | None:
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt == retries - 1:
                print(f"[WARN] Failed {url}: {e}")
                return None
            time.sleep(1)


def fetch_item(item_id: int) -> dict | None:
    data = fetch_json(f"{BASE_URL}/item/{item_id}.json")
    if not data or data.get("deleted") or data.get("dead"):
        return None
    return {
        "id": data.get("id"),
        "type": data.get("type", ""),
        "title": data.get("title", ""),
        "text": data.get("text", ""),
        "url": data.get("url", ""),
        "score": data.get("score", 0),
        "author": data.get("by", ""),
        "comments": data.get("descendants", 0),
        "timestamp": data.get("time", 0),
        "datetime": datetime.utcfromtimestamp(data.get("time", 0)).strftime("%Y-%m-%d %H:%M:%S") if data.get("time") else "",
    }


def classify_stories(df: pd.DataFrame) -> pd.DataFrame:
    df["combined_text"] = df["title"].fillna("") + " " + df["text"].fillna("")
    df["clean_text"] = df["combined_text"].apply(clean_text)
    
    df_valid = df[df["clean_text"] != ""].copy()
    df_empty = df[df["clean_text"] == ""].copy()
    
    if len(df_valid) > 10:
        # TF-IDF
        tfidf = TfidfVectorizer(max_features=1000)
        X = tfidf.fit_transform(df_valid["clean_text"])
        
        y = (df_valid["score"] > 200).astype(int)
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X, y)
        
        y_pred = model.predict(X)
        df_valid["category"] = ["Hot" if p == 1 else "Normal" for p in y_pred]
    else:
        df_valid["category"] = "Normal"
    
    df_empty["category"] = "Uncategorized"
    df_result = pd.concat([df_valid, df_empty], ignore_index=True)
    df_result = df_result.drop(columns=["combined_text", "clean_text"])
    
    return df_result

def clean_text(text: str) -> str:
    """Text cleaning: remove HTML tags, punctuation, stop words, and lowercase."""
    if not text: 
        return ""
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    stop_words = set(stopwords.words("english"))
    words = text.split()
    clean_words = [word for word in words if word not in stop_words and len(word) > 2]
    return " ".join(clean_words)

def scrape_stories(endpoint: str, limit: int = 500) -> list[dict]:
    print(f"[INFO] Fetching story IDs from /{endpoint} ...")
    story_ids = fetch_json(f"{BASE_URL}/{endpoint}.json")
    if not story_ids:
        return []

    story_ids = story_ids[:limit]
    print(f"[INFO] Fetching {len(story_ids)} stories concurrently ...")

    results = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(fetch_item, sid): sid for sid in story_ids}
        for i, future in enumerate(as_completed(futures)):
            item = future.result()
            if item:
                results.append(item)
            if (i + 1) % 100 == 0:
                print(f"  ... {i + 1}/{len(story_ids)} done")

    return results


def main():
    all_stories = []

    # Top stories (general tech news)
    top = scrape_stories("topstories", limit=300)
    print(f"[INFO] Top stories: {len(top)}")
    all_stories.extend(top)

    # Ask HN stories (community discussions)
    ask = scrape_stories("askstories", limit=200)
    print(f"[INFO] Ask HN stories: {len(ask)}")
    all_stories.extend(ask)

    # Deduplicate by id
    seen = set()
    unique = []
    for s in all_stories:
        if s["id"] not in seen:
            seen.add(s["id"])
            unique.append(s)

    print(f"[INFO] Total unique records: {len(unique)}")

    df = pd.DataFrame(unique)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    print("[INFO] Classifying stories ...")
    df = classify_stories(df)

    csv_path = os.path.join(OUTPUT_DIR, "hn_stories.csv")
    json_path = os.path.join(OUTPUT_DIR, "hn_stories.json")

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_json(json_path, orient="records", indent=2, force_ascii=False)

    print(f"[DONE] Saved {len(df)} records")
    print(f"  CSV  → {csv_path}")
    print(f"  JSON → {json_path}")
    print("\nSample (top 5 by score):")
    print(df[["title", "score", "author", "comments"]].head())


if __name__ == "__main__":
    main()
