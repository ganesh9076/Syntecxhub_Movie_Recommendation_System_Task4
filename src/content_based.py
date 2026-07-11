# %%
"""
content_based.py
-----------------
Content-based movie recommender.

Builds a TF-IDF representation of each movie's "metadata soup"
(genres + aggregated user tags) and recommends movies whose soup is
most similar (cosine similarity) to a movie the user already likes.

This does NOT use other users' ratings at all -> it works fine even for
brand-new / rarely-rated movies (no cold-start problem for items), and
recommendations are easy to explain ("similar genres/tags to X").
"""

import os
import sys
import difflib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_prep import load_clean_data


class ContentBasedRecommender:
    def __init__(self, movies: pd.DataFrame):
        self.movies = movies.reset_index(drop=True)
        self._title_to_idx = pd.Series(
            self.movies.index, index=self.movies["clean_title"].str.lower()
        )
        self.vectorizer = None
        self.tfidf_matrix = None
        self.similarity = None

    def fit(self):
        self.vectorizer = TfidfVectorizer(
            token_pattern=r"[a-zA-Z0-9']+",  
            min_df=1,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.movies["soup"])
        # cosine similarity between every pair of movies
        self.similarity = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        return self

    def _resolve_title(self, title: str):
        """Find the closest matching movie title (case-insensitive, fuzzy)."""
        key = title.lower().strip()
        if key in self._title_to_idx:
            return self._title_to_idx[key] if np.isscalar(self._title_to_idx[key]) \
                else self._title_to_idx[key].iloc[0]

        choices = list(self._title_to_idx.index)
        close = difflib.get_close_matches(key, choices, n=1, cutoff=0.5)
        if close:
            return self._title_to_idx[close[0]]

        # fall back to substring search
        contains = self.movies[self.movies["clean_title"].str.lower().str.contains(key, na=False)]
        if len(contains):
            return contains.index[0]

        return None

    def recommend(self, title: str, top_n: int = 10) -> pd.DataFrame:
        idx = self._resolve_title(title)
        if idx is None:
            raise ValueError(f"Could not find a movie matching '{title}' in the catalog.")

        sims = list(enumerate(self.similarity[idx]))
        sims = sorted(sims, key=lambda x: x[1], reverse=True)
        sims = [s for s in sims if s[0] != idx][:top_n]

        result_idx = [i for i, _ in sims]
        scores = [round(float(s), 4) for _, s in sims]

        out = self.movies.iloc[result_idx][["movieId", "clean_title", "year", "genres"]].copy()
        out["similarity"] = scores
        out.insert(0, "matched_query_title", self.movies.iloc[idx]["clean_title"])
        return out.reset_index(drop=True)


def build_and_save(save_path=None):
    import pickle
    movies, ratings, links = load_clean_data()
    model = ContentBasedRecommender(movies).fit()
    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(model, f)
    return model


if __name__ == "__main__":
    movies, ratings, links = load_clean_data()
    model = ContentBasedRecommender(movies).fit()

    for query in ["Toy Story", "The Matrix", "The Dark Knight"]:
        print(f"\n=== Because you liked: {query} ===")
        print(model.recommend(query, top_n=5).to_string(index=False))
