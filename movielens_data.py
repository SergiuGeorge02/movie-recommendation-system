import pandas as pd

# Path to MovieLens dataset folder
folder = r"C:\Users\Sergiu\Desktop\Big Data project\movies"

#Data loading here
movies = pd.read_csv(f"{folder}\\movie.csv")
genome_tags = pd.read_csv(f"{folder}\\genome_tags.csv")
genome_scores = pd.read_csv(f"{folder}\\genome_scores.csv")
ratings = pd.read_csv(f"{folder}\\rating.csv")
user_tags = pd.read_csv(f"{folder}\\tag.csv")


tagged_scores = genome_scores.merge(genome_tags, on="tagId")
top_tags = (
    tagged_scores.sort_values(["movieId", "relevance"], ascending=False)
    .groupby("movieId")
    .head(10)
)
movie_tags = (
    top_tags.groupby("movieId")["tag"]
    .apply(lambda tags: ", ".join(tags))
    .reset_index()
    .rename(columns={"tag": "genome_summary"})
)

user_tags_grouped = (
    user_tags.groupby("movieId")["tag"]
    .apply(lambda tags: ", ".join(str(tag) for tag in set(tags) if pd.notna(tag)))
    .reset_index()
    .rename(columns={"tag": "user_summary"})
)

avg_ratings = ratings.groupby("movieId")["rating"].mean().reset_index()
avg_ratings = avg_ratings.rename(columns={"rating": "rating_value"})

#here I am merging everything into the dataset
movies = movies.merge(movie_tags, on="movieId", how="left")
movies = movies.merge(user_tags_grouped, on="movieId", how="left")
movies = movies.merge(avg_ratings, on="movieId", how="left")

#here i am combining the genome sumary with the user summary
movies["summary"] = movies[["genome_summary", "user_summary"]].fillna("").agg(" ".join, axis=1).str.strip()

# Here I am exracting the year
movies["release_year"] = movies["title"].str.extract(r"\((\d{4})\)").astype(float)
movies["title"] = movies["title"].str.replace(r"\(\d{4}\)", "", regex=True).str.strip()

movies["type"] = "movie"
movies_cleaned = movies[["title", "genres", "summary", "rating_value", "release_year", "type"]]

output_path = "cleaned_movies.csv"
movies_cleaned.to_csv(output_path, index=False)
print(f"Saved enhanced cleaned_movies.csv to:\n{output_path}")
