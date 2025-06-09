import pandas as pd
import requests
from urllib.parse import quote
import time

TMDB_API_KEY = "071525ace6ccc5a12c896c2fed2b628e"

# Function to fetch ratings
def fetch_tmdb_rating(title, media_type="movie"):
    try:
        url = f"https://api.themoviedb.org/3/search/{media_type}"
        params = {"api_key": TMDB_API_KEY, "query": title}
        response = requests.get(url, params=params).json()

        if response.get("results"):
            rating = response["results"][0].get("vote_average")
            if rating:
                return round(float(rating), 1)
    except Exception as e:
        print(f"❌ Rating fetch failed for '{title}': {e}")
    return None

# Function to fetch the summary
def fetch_tmdb_summary(title, media_type="movie"):
    try:
        url = f"https://api.themoviedb.org/3/search/{media_type}"
        params = {"api_key": TMDB_API_KEY, "query": title}
        response = requests.get(url, params=params).json()

        if response.get("results"):
            overview = response["results"][0].get("overview")
            if overview:
                return overview.strip()
    except Exception as e:
        print(f"❌ Summary fetch failed for '{title}': {e}")
    return None

# === Load dataset ===
df = pd.read_csv("dataset.csv")
#
# # === Handle missing ratings ===
# missing_ratings = df[df["rating_value"].isna()]
# print(f"🔍 Missing ratings: {len(missing_ratings)}")
#
# for idx, row in missing_ratings.iterrows():
#     title = row["title"]
#     media_type = "tv" if row["type"].lower() == "tv_show" else "movie"
#     rating = fetch_tmdb_rating(title, media_type)
#     if rating:
#         df.at[idx, "rating_value"] = rating
#         print(f"✅ Filled rating for '{title}': {rating}")
#     else:
#         print(f"⚠️ No rating found for '{title}'")
#     time.sleep(0.25)

# === Handle missing summaries ===
missing_summaries = df[df["summary"].isna()]
print(f"\n🔍 Missing summaries: {len(missing_summaries)}")

for idx, row in missing_summaries.iterrows():
    title = row["title"]
    media_type = "tv" if row["type"].lower() == "tv_show" else "movie"
    summary = fetch_tmdb_summary(title, media_type)
    if summary:
        df.at[idx, "summary"] = summary
        print(f"✅ Filled summary for '{title}'")
    else:
        print(f"⚠️ No summary found for '{title}'")
    time.sleep(0.25)

# === Save updated dataset ===
df.to_csv("dataset_filled.csv", index=False)
print("\n💾 Saved updated dataset to 'dataset_filled.csv'")
