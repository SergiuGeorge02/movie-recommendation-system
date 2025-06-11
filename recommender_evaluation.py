
import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------- Evaluation Function ----------------------

def evaluate_recommender(df, user_genres, min_year, min_rating, keywords, top_n=10):
    df["genres"] = df["genres"].fillna("")
    df["summary"] = df["summary"].fillna("")
    df["content"] = df["genres"] + " " + df["summary"]

    filtered_df = df[
        (df["release_year"] >= min_year) &
        (df["rating_value"] >= min_rating) &
        (df["genres"].str.lower().apply(lambda g: any(genre in g.lower() for genre in user_genres)))
    ].copy()

    if filtered_df.empty:
        return pd.DataFrame()

    user_profile_text = " ".join(user_genres + keywords)

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(filtered_df["content"].tolist() + [user_profile_text])

    similarity_scores = cosine_similarity(tfidf_matrix[:-1], tfidf_matrix[-1])
    filtered_df["similarity"] = similarity_scores.flatten()

    # More flexible keyword matching (partial match)
    filtered_df["keyword_matches"] = filtered_df["summary"].apply(
        lambda text: sum(1 for kw in keywords if kw.lower() in text.lower())
    )

    ranked_df = filtered_df.sort_values(
        by=["similarity", "keyword_matches", "rating_value"],
        ascending=False
    ).head(top_n)

    return ranked_df

# ---------------------- Batch Evaluation ----------------------

def evaluate_recommender_batch(df, user_profiles, top_n=10):
    profile_results = []
    total_matches = 0
    total_possible = 0

    for i, profile in enumerate(user_profiles):
        recommendations = evaluate_recommender(
            df,
            user_genres=profile["user_genres"],
            min_year=profile["min_year"],
            min_rating=profile["min_rating"],
            keywords=profile["keywords"],
            top_n=top_n
        )

        matches = recommendations["keyword_matches"].sum()
        possible = len(profile["keywords"]) * len(recommendations)

        profile_results.append({
            "profile_id": i + 1,
            "matched_keywords": matches,
            "total_possible": possible,
            "precision": matches / possible if possible else 0
        })

        total_matches += matches
        total_possible += possible

    overall_precision = total_matches / total_possible if total_possible else 0
    return overall_precision, profile_results

# ---------------------- Plotting ----------------------

def plot_precision_scores(profile_eval, title="Precision@K per User Profile"):
    profile_ids = [res["profile_id"] for res in profile_eval]
    precisions = [res["precision"] for res in profile_eval]

    plt.figure(figsize=(8, 5))
    plt.bar(profile_ids, precisions, edgecolor='black')
    plt.title(title)
    plt.xlabel("User Profile ID")
    plt.ylabel("Precision@K")
    plt.ylim(0, 1)
    plt.xticks(profile_ids)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

# ---------------------- Main Execution ----------------------

if __name__ == "__main__":
    dataset_path = "dataset.csv"
    df = pd.read_csv(dataset_path)

    test_profiles = [
        {"user_genres": ["drama", "thriller"], "min_year": 2015, "min_rating": 7.0, "keywords": ["mystery", "revenge"]},
        {"user_genres": ["comedy"], "min_year": 2010, "min_rating": 6.0, "keywords": ["friendship", "family"]},
        {"user_genres": ["sci-fi", "action"], "min_year": 2018, "min_rating": 7.5, "keywords": ["future", "technology"]},
        {"user_genres": ["fantasy"], "min_year": 2000, "min_rating": 7.0, "keywords": ["magic", "quest"]},
        {"user_genres": ["romance", "drama"], "min_year": 2012, "min_rating": 6.5, "keywords": ["love", "betrayal"]}
    ]

    overall_precision, profile_eval = evaluate_recommender_batch(df, test_profiles, top_n=5)

    print(f"\n🔍 Overall Precision@5: {overall_precision:.2f}")
    for result in profile_eval:
        print(f"Profile {result['profile_id']}: Precision = {result['precision']:.2f} "
              f"(Matched {result['matched_keywords']} of {result['total_possible']})")

    plot_precision_scores(profile_eval)
