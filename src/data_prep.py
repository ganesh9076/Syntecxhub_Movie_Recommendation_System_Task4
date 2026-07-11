# %%
"""
data_prep.py
------------
Loads the raw MovieLens (ml-latest-small) CSVs, cleans them, and builds the
"metadata soup" text feature used by the content-based recommender.

Dataset: MovieLens ml-latest-small (9,742 movies / 100,836 ratings / 610 users / 3,683 tags)
Source : https://grouplens.org/datasets/movielens/

Run directly to sanity-check the cleaning steps, or import `load_clean_data()`
from other scripts.
"""

import os
import re
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _extract_year(title: str):
    """Pull a 4-digit release year out of a MovieLens title like
    'Toy Story (1995)'. Returns None if no year is present."""
    match = re.search(r"\((\d{4})\)\s*$", title.strip())
    return int(match.group(1)) if match else None


def _clean_title(title: str):
    """Strip the trailing '(YYYY)' from a title, and fix the
    'Matrix, The (1999)' -> 'The Matrix (1999)' style inversion
    that MovieLens uses for sorting."""
    title = re.sub(r"\s*\(\d{4}\)\s*$", "", title).strip()
    m = re.match(r"^(.*),\s*(The|A|An)$", title)
    if m:
        title = f"{m.group(2)} {m.group(1)}"
    return title


def load_raw():
    movies = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))
    ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
    tags = pd.read_csv(os.path.join(DATA_DIR, "tags.csv"))
    links = pd.read_csv(os.path.join(DATA_DIR, "links.csv"))
    return movies, ratings, tags, links


def clean_movies(movies: pd.DataFrame) -> pd.DataFrame:
    movies = movies.copy()

    # Drop exact duplicate rows / duplicate movieIds, if any slipped in
    movies = movies.drop_duplicates(subset="movieId").reset_index(drop=True)

    # Year + display title
    movies["year"] = movies["title"].apply(_extract_year)
    movies["clean_title"] = movies["title"].apply(_clean_title)

    # genres: '(no genres listed)' -> empty, pipe-separated -> list
    movies["genres"] = movies["genres"].replace("(no genres listed)", "")
    movies["genre_list"] = movies["genres"].apply(
        lambda g: [x.strip() for x in g.split("|") if x.strip()]
    )

    # A version of genres with no separators/spaces, good for text vectorizing
    # e.g. "Sci-Fi" -> "scifi" so it isn't split into two tokens
    movies["genres_tokens"] = movies["genre_list"].apply(
        lambda lst: " ".join(g.lower().replace("-", "").replace(" ", "") for g in lst)
    )

    return movies


def clean_tags(tags: pd.DataFrame) -> pd.DataFrame:
    tags = tags.copy()
    tags["tag"] = tags["tag"].astype(str).str.strip().str.lower()
    tags = tags[tags["tag"] != ""]
    return tags


def build_tag_soup(tags: pd.DataFrame) -> pd.DataFrame:
    """Aggregate all user tags per movie into one space-joined string."""
    agg = (
        tags.groupby("movieId")["tag"]
        .apply(lambda s: " ".join(t.replace(" ", "") for t in s))
        .reset_index()
        .rename(columns={"tag": "tags_soup"})
    )
    return agg


def build_metadata_soup(movies: pd.DataFrame, tags: pd.DataFrame) -> pd.DataFrame:
    """Combine genres + user tags into a single text column ('soup') per
    movie, used as the input to TF-IDF for content-based filtering.
    Genres are repeated 3x so they carry more weight than the sparser tags."""
    movies = clean_movies(movies)
    tag_soup = build_tag_soup(clean_tags(tags))

    df = movies.merge(tag_soup, on="movieId", how="left")
    df["tags_soup"] = df["tags_soup"].fillna("")

    df["soup"] = (
        ((df["genres_tokens"] + " ") * 3) + df["tags_soup"]
    ).str.strip()

    return df


def clean_ratings(ratings: pd.DataFrame, valid_movie_ids=None) -> pd.DataFrame:
    ratings = ratings.copy()
    ratings = ratings.drop_duplicates(subset=["userId", "movieId"])
    ratings = ratings.dropna(subset=["userId", "movieId", "rating"])
    ratings = ratings[(ratings["rating"] >= 0.5) & (ratings["rating"] <= 5.0)]
    if valid_movie_ids is not None:
        ratings = ratings[ratings["movieId"].isin(valid_movie_ids)]
    return ratings


def load_clean_data():
    """One-call convenience function returning everything downstream code needs."""
    movies_raw, ratings_raw, tags_raw, links = load_raw()
    movies = build_metadata_soup(movies_raw, tags_raw)
    ratings = clean_ratings(ratings_raw, valid_movie_ids=set(movies["movieId"]))
    return movies, ratings, links


if __name__ == "__main__":
    movies, ratings, links = load_clean_data()
    print("Movies:", movies.shape)
    print("Ratings:", ratings.shape)
    print(movies[["movieId", "clean_title", "year", "genres", "soup"]].head(8).to_string())
