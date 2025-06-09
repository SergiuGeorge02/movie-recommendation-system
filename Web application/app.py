from flask import Flask, render_template, request, session, redirect, url_for
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import re
import requests

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# TMDb API Key
TMDB_API_KEY = "071525ace6ccc5a12c896c2fed2b628e"

# Hear spacy is loaded
nlp = spacy.load("en_core_web_sm")

# dataset is being loaded here.
df = pd.read_csv("C:/Users/Sergiu/Desktop/Big Data Project/dataset/dataset.csv")
df["summary"] = df["summary"].fillna("")
df["genres"] = df["genres"].fillna("")
df["rating_value"] = df["rating_value"].fillna(0)
df["release_year"] = df["release_year"].fillna(0)
df["content"] = df["genres"] + " " + df["summary"]
df = df.dropna(subset=["title"])

# Here the questions for the users are introduced
questions = [
    {"key": "type", "text": "🎬 If you were to choose you would choose movies or TV shows?"},
    {"key": "genres", "text": "🎭 What genres do you enjoy? (e.g., 'comedy, action')"},
    {"key": "release_year", "text": "📅 What's the earliest release year you're interested in?"},
    {"key": "rating_value", "text": "⭐ What's your minimum rating? (e.g., 'at least 7')"},
    {"key": "favorite", "text": "💬 Name a movie or TV show you liked recently:"}
]

all_genres = ["drama", "comedy", "sci-fi", "thriller", "romance", "horror",
              "action", "crime", "fantasy", "adventure", "mystery", "documentary"]

#function that will extract preferences, genres and years which are introduced by the users
def extract_preferences(text):
    document = nlp(text.lower())
    genres_preferences = []
    year = None
    rating = None

    for token in document:
        if token.text in all_genres:
            genres_preferences.append(token.text)

    year_match = re.search(r"\b(19|20)\d{2}\b", text)
    if year_match:
        year = int(year_match.group())

    rating_match = re.search(r"(?:at least|over|rating of)?\s*(\d+(\.\d+)?)", text)
    if rating_match:
        rating = float(rating_match.group(1))

    return {
        "genres": ", ".join(genres_preferences) if genres_preferences else None,
        "release_year": year,
        "rating_value": rating
    }

#this will fetch the poster into the website
def fetch_poster_tmdb(title, media_type="movie"):
    try:
        tmdb_type = "tv" if media_type.lower() == "tv_show" else "movie"
        search_url = f"https://api.themoviedb.org/3/search/{tmdb_type}"
        params = {
            "api_key": TMDB_API_KEY,
            "query": title
        }
        response = requests.get(search_url, params=params).json()
        print(f"Searching TMDb for: {title} [{tmdb_type}]")

        if response.get("results"):
            poster_path = response["results"][0].get("poster_path")
            if poster_path:
                full_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
                print(f"Poster found: {full_url}")
                return full_url
            else:
                print("No poster_path found.")
        else:
            print("TMDb returned no results.")
    except Exception as e:
        print("TMDb error:", e)

    return "https://via.placeholder.com/100x150?text=No+Image"

@app.route("/", methods=["GET", "POST"])
def chat():
    if request.method == "GET" and "qa_pairs" not in session:
        session.clear()
        session["step"] = 0
        session["qa_pairs"] = []

    step = session.get("step", 0)
    qa_pairs = session.get("qa_pairs", [])

    if request.method == "POST":
        user_input = request.form["user_input"].strip()

        if step in [1, 2, 3]:
            extracted = extract_preferences(user_input)

            if step == 1 and not extracted["genres"]:
                return render_template("chat.html", qa_pairs=qa_pairs, question=questions[step]["text"],
                                       error="❌ Please mention a genre like 'comedy' or 'action'.")
            if step == 2 and not extracted["release_year"]:
                return render_template("chat.html", qa_pairs=qa_pairs, question=questions[step]["text"],
                                       error="❌ Try something like 'after 2010'.")
            if step == 3 and extracted["rating_value"] is None:
                return render_template("chat.html", qa_pairs=qa_pairs, question=questions[step]["text"],
                                       error="❌ Say something like 'at least 7' or just '8'.")

            if step == 1:
                user_input = extracted["genres"]
            elif step == 2:
                user_input = str(extracted["release_year"])
            elif step == 3:
                user_input = str(extracted["rating_value"])

        qa_pairs.append({
            "question": questions[step]["text"],
            "answer": user_input
        })
        session["qa_pairs"] = qa_pairs
        session["step"] = step + 1

        if session["step"] >= len(questions):
            return redirect(url_for("results"))

    step = session.get("step", 0)
    qa_pairs = session.get("qa_pairs", [])
    next_question = None
    if step < len(questions) and len(qa_pairs) == step:
        next_question = questions[step]["text"]

    return render_template("chat.html", qa_pairs=qa_pairs, question=next_question)

@app.route("/results")
def results():
    if "qa_pairs" not in session or len(session["qa_pairs"]) < len(questions):
        return redirect(url_for("chat"))

    prefs = {
        questions[i]["key"]: session["qa_pairs"][i]["answer"]
        for i in range(len(questions))
    }

    print("\n🔎 Preferences collected:")
    for k, v in prefs.items():
        print(f"- {k}: {v}")

    filtered = df.copy()

    # Normalize type
    user_type = prefs["type"].strip().lower()
    if "tv" in user_type:
        user_type = "tv_show"
    elif "movie" in user_type:
        user_type = "movie"

    filtered = filtered[filtered["type"].str.lower() == user_type]
    print(f"After type filter ({user_type}):", len(filtered))

    genre_str = prefs["genres"].lower()
    filtered = filtered[filtered["genres"].fillna("").str.lower().str.contains(genre_str)]
    print("After genre filter:", len(filtered))

    try:
        year = int(prefs["release_year"])
        filtered = filtered[filtered["release_year"] >= year]
        print(f"After release year ≥ {year}:", len(filtered))
    except:
        print("Invalid release_year")

    try:
        rating = float(prefs["rating_value"])
        filtered = filtered[filtered["rating_value"] >= rating]
        print(f"After rating ≥ {rating}:", len(filtered))
    except:
        print("Invalid rating_value")

    if filtered.empty:
        print("No results after filtering.")
        session.clear()
        return render_template("chat.html", results=[], error="❌ No results match your preferences.")

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(filtered["content"])

    fav_title = prefs["favorite"].strip().lower()
    match = df[df["title"].str.lower().str.strip() == fav_title]

    if not match.empty:
        fav_index = match.index[0]
        fav_vec = tfidf.transform([df.loc[fav_index, "content"]])
        cosine_sim = cosine_similarity(fav_vec, tfidf_matrix)
        filtered["score"] = cosine_sim.flatten()
        recommended = filtered.sort_values("score", ascending=False).head(5)
        print("Using favorite for similarity-based recommendations.")
    else:
        recommended = filtered.sample(min(5, len(filtered)))
        print("Favorite not found. Showing random relevant matches.")

    # 🔗 Fetch posters from TMDb
    recommended["image_url"] = recommended["title"].apply(
        lambda title: fetch_poster_tmdb(title, media_type=user_type)
    )

    session.clear()
    return render_template("chat.html", results=recommended[["title", "genres", "rating_value", "summary", "image_url"]].to_dict(orient="records"))

@app.route("/reset")
def reset():
    session.clear()
    return redirect(url_for("chat"))

if __name__ == "__main__":
    app.run(debug=True)
