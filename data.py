import json
import pandas as pd
from bs4 import BeautifulSoup

file = "data.ndjson"

# Load NDJSON lines into a list of dictionaries
with open(file, "r", encoding="utf-8") as frame:
    data = [json.loads(line) for line in frame]

# Convert to DataFrame
df = pd.DataFrame(data)

# Define columns to extract
title = "name"
genre= "genres"
description = "summary"
rating = "rating"
date = "premiered"

# Select only needed columns
df = df[[title, genre, description, rating, date]]

# Drop rows missing title or genres
df = df.dropna(subset=[title, genre])

# Clean HTML from summary/description
def clean_html(text):
    return BeautifulSoup(text, "html.parser").get_text() if isinstance(text, str) else ""

df[description] = df[description].apply(clean_html)

# Extract average rating
df["rating_value"] = df[rating].apply(lambda r: r.get("average") if isinstance(r, dict) else None)

# Drop original rating column
df = df.drop(columns=[rating])

# Convert genres to comma-separated string
df[genre] = df[genre].apply(lambda g: ", ".join(g) if isinstance(g, list) else g)

# Extract release year
df["release_year"] = pd.to_datetime(df[date], errors="coerce").dt.year
df = df.drop(columns=[date])


df["type"] = "tv_show"

# Preview cleaned DataFrame
print(df.head())

# Save to CSV
df.to_csv("cleaned_tv_shows.csv", index=False)
print("Cleaned dataset saved as tv_shows.csv")
