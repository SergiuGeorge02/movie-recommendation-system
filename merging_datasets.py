import pandas as pd

# Loading the cleaned TV shows and movies datasets
tv_df = pd.read_csv("cleaned_tv_shows.csv")
movie_df = pd.read_csv("cleaned_movies.csv")

# Rename 'name' column in tv_df to 'title' for consistency
tv_df = tv_df.rename(columns={"name": "title"})

# Ensure both DataFrames have the same column order
tv_df = tv_df[["title", "genres", "summary", "rating_value", "release_year", "type"]]
movie_df = movie_df[["title", "genres", "summary", "rating_value", "release_year", "type"]]

# Concatenate the two DataFrames
combined_df = pd.concat([tv_df, movie_df], ignore_index=True)

# Save the combined DataFrame to a new CSV file
combined_df.to_csv("dataset.csv", index=False)
print("✅ Saved combined dataset as combined_movies_and_tv.csv")
