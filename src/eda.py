# %%
"""
eda.py
------
Exploratory Data Analysis on the MovieLens ml-latest-small dataset.
Produces summary stats to stdout and saves plots under outputs/figures/.

Run: python src/eda.py   (from the project root)
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_prep import load_raw, clean_movies, clean_ratings, clean_tags

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["axes.grid"] = True


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def run_eda():
    movies_raw, ratings_raw, tags_raw, links = load_raw()

    section("1. RAW SHAPES")
    print(f"movies : {movies_raw.shape}")
    print(f"ratings: {ratings_raw.shape}")
    print(f"tags   : {tags_raw.shape}")
    print(f"links  : {links.shape}")

    section("2. MISSING VALUES (raw)")
    print("movies:\n", movies_raw.isna().sum())
    print("ratings:\n", ratings_raw.isna().sum())
    print("tags:\n", tags_raw.isna().sum())

    section("3. DUPLICATES (raw)")
    print("duplicate movieIds:", movies_raw["movieId"].duplicated().sum())
    print("duplicate (userId, movieId) rating pairs:",
          ratings_raw.duplicated(subset=["userId", "movieId"]).sum())

    # Cleaned versions for downstream analysis
    movies = clean_movies(movies_raw)
    ratings = clean_ratings(ratings_raw, valid_movie_ids=set(movies["movieId"]))
    tags = clean_tags(tags_raw)

    section("4. AFTER CLEANING")
    print(f"movies : {movies.shape} (dropped {movies_raw.shape[0] - movies.shape[0]} dup rows)")
    print(f"ratings: {ratings.shape} (dropped {ratings_raw.shape[0] - ratings.shape[0]} bad/dup rows)")
    print("movies with no genre listed:", (movies["genres"] == "").sum())
    print("movies with a parsed year:", movies["year"].notna().sum(), "/", len(movies))

    section("5. RATING DISTRIBUTION")
    print(ratings["rating"].describe())
    ratings["rating"].value_counts().sort_index().plot(kind="bar", color="#4C72B0")
    plt.title("Distribution of Ratings")
    plt.xlabel("Rating (stars)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "rating_distribution.png"), dpi=120)
    plt.close()

    section("6. RATINGS PER USER / PER MOVIE (sparsity)")
    ratings_per_user = ratings.groupby("userId").size()
    ratings_per_movie = ratings.groupby("movieId").size()
    print("Ratings per user  -> min:", ratings_per_user.min(),
          "median:", ratings_per_user.median(), "max:", ratings_per_user.max())
    print("Ratings per movie -> min:", ratings_per_movie.min(),
          "median:", ratings_per_movie.median(), "max:", ratings_per_movie.max())

    n_users = ratings["userId"].nunique()
    n_movies = movies["movieId"].nunique()
    n_ratings = len(ratings)
    sparsity = 1 - n_ratings / (n_users * n_movies)
    print(f"User-item matrix sparsity: {sparsity:.4%}  "
          f"({n_users} users x {n_movies} movies, {n_ratings} known ratings)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(ratings_per_user, bins=50, color="#55A868")
    axes[0].set_title("Ratings per User")
    axes[0].set_xlabel("# ratings given")
    axes[1].hist(ratings_per_movie, bins=50, color="#C44E52")
    axes[1].set_title("Ratings per Movie")
    axes[1].set_xlabel("# ratings received")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "ratings_sparsity.png"), dpi=120)
    plt.close()

    section("7. GENRE FREQUENCY")
    genre_counts = movies.explode("genre_list")["genre_list"].value_counts()
    genre_counts = genre_counts[genre_counts.index != ""]
    print(genre_counts)
    genre_counts.plot(kind="barh", color="#8172B2")
    plt.title("Movie Count by Genre")
    plt.xlabel("Number of Movies")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "genre_counts.png"), dpi=120)
    plt.close()

    section("8. MOVIES PER RELEASE YEAR")
    by_year = movies.dropna(subset=["year"]).groupby("year").size()
    by_year.plot(color="#DD8452")
    plt.title("Movies Released per Year")
    plt.xlabel("Year")
    plt.ylabel("Number of Movies")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "movies_per_year.png"), dpi=120)
    plt.close()

    section("9. AVERAGE RATING vs POPULARITY (Bayesian-adjusted)")
    stats = ratings.groupby("movieId")["rating"].agg(["mean", "count"])
    C = stats["count"].mean()
    m = stats["mean"].mean()
    stats["weighted_rating"] = (stats["count"] / (stats["count"] + C)) * stats["mean"] + \
                                (C / (stats["count"] + C)) * m
    top = stats.sort_values("weighted_rating", ascending=False).head(10)
    top = top.join(movies.set_index("movieId")["clean_title"])
    print(top[["clean_title", "mean", "count", "weighted_rating"]].to_string())

    print(f"\nAll figures saved to: {FIG_DIR}")
    return movies, ratings


if __name__ == "__main__":
    run_eda()
