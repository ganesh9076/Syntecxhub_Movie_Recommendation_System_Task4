# %%
"""
collaborative.py
-----------------
Collaborative filtering recommenders, built purely from the user-item
ratings matrix (no metadata used at all).

Two variants:
  1. ItemBasedCF      - "users who liked movie X also liked movie Y"
                        (cosine similarity between movie rating-vectors)
  2. MatrixFactorizationCF - learns latent user/item factors with Truncated
                        SVD on the mean-centered rating matrix, then
                        predicts a rating for every unseen movie for a user
                        and recommends the highest-predicted ones.

Both only need ratings.csv - genres/tags are irrelevant here, which is the
key difference from the content-based approach in content_based.py.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_prep import load_clean_data


class _RatingMatrixMixin:
    def _build_matrix(self, ratings: pd.DataFrame, movies: pd.DataFrame):
        self.user_ids = np.sort(ratings["userId"].unique())
        self.movie_ids = np.sort(movies["movieId"].unique())

        self.user_id_to_idx = {u: i for i, u in enumerate(self.user_ids)}
        self.movie_id_to_idx = {m: i for i, m in enumerate(self.movie_ids)}
        self.idx_to_movie_id = {i: m for m, i in self.movie_id_to_idx.items()}

        rows = ratings["userId"].map(self.user_id_to_idx).values
        cols = ratings["movieId"].map(self.movie_id_to_idx).values
        vals = ratings["rating"].values

        self.matrix = csr_matrix(
            (vals, (rows, cols)),
            shape=(len(self.user_ids), len(self.movie_ids)),
        )

        self.movie_id_to_title = movies.set_index("movieId")["clean_title"].to_dict()
        self.title_to_movie_id = {
            t.lower(): mid for mid, t in self.movie_id_to_title.items()
        }


class ItemBasedCF(_RatingMatrixMixin):
    """Recommend movies similar to a given movie, based on rating patterns
    (not metadata): two movies are 'similar' if the same users tended to
    rate them similarly."""

    def __init__(self, ratings: pd.DataFrame, movies: pd.DataFrame, min_ratings=5):
        self.ratings_raw = ratings
        self.movies = movies
        self.min_ratings = min_ratings

    def fit(self):
        # Only keep movies with enough ratings for a stable similarity estimate
        counts = self.ratings_raw.groupby("movieId").size()
        keep_ids = counts[counts >= self.min_ratings].index
        movies_f = self.movies[self.movies["movieId"].isin(keep_ids)]
        ratings_f = self.ratings_raw[self.ratings_raw["movieId"].isin(keep_ids)]

        self._build_matrix(ratings_f, movies_f)
        # item-item similarity = cosine similarity between columns (movies)
        self.item_similarity = cosine_similarity(self.matrix.T)
        return self

    def recommend(self, title: str, top_n: int = 10) -> pd.DataFrame:
        key = title.lower().strip()
        movie_id = self.title_to_movie_id.get(key)
        if movie_id is None:
            matches = [t for t in self.title_to_movie_id if key in t]
            if not matches:
                raise ValueError(f"'{title}' not found among movies with >= "
                                  f"{self.min_ratings} ratings.")
            movie_id = self.title_to_movie_id[matches[0]]

        idx = self.movie_id_to_idx[movie_id]
        sims = list(enumerate(self.item_similarity[idx]))
        sims = sorted(sims, key=lambda x: x[1], reverse=True)
        sims = [s for s in sims if s[0] != idx][:top_n]

        rows = []
        for i, score in sims:
            mid = self.idx_to_movie_id[i]
            rows.append({
                "movieId": mid,
                "clean_title": self.movie_id_to_title[mid],
                "similarity": round(float(score), 4),
            })
        out = pd.DataFrame(rows)
        out.insert(0, "matched_query_title", self.movie_id_to_title[movie_id])
        return out


class MatrixFactorizationCF(_RatingMatrixMixin):
    """Learns latent factors for users & movies (Truncated SVD on the
    mean-centered rating matrix) and predicts ratings for unseen movies,
    i.e. classic latent-factor collaborative filtering."""

    def __init__(self, ratings: pd.DataFrame, movies: pd.DataFrame, n_factors=30):
        self.ratings_raw = ratings
        self.movies = movies
        self.n_factors = n_factors

    def fit(self):
        self._build_matrix(self.ratings_raw, self.movies)
        dense = self.matrix.toarray()

        # mean-center each user's ratings (ignore zeros=unrated when computing mean)
        user_means = np.true_divide(
            dense.sum(axis=1), (dense != 0).sum(axis=1),
            out=np.zeros(dense.shape[0]), where=(dense != 0).sum(axis=1) != 0
        )
        self.user_means = user_means
        centered = dense - user_means.reshape(-1, 1)
        centered[dense == 0] = 0  # keep unrated entries at 0 after centering

        self.svd = TruncatedSVD(n_components=min(self.n_factors, min(dense.shape) - 1),
                                 random_state=42)
        self.user_factors = self.svd.fit_transform(centered)
        self.item_factors = self.svd.components_.T
        self._dense_known = dense  # to mask out already-rated movies
        return self

    def predict_ratings_for_user(self, user_id) -> np.ndarray:
        u_idx = self.user_id_to_idx[user_id]
        preds = self.user_factors[u_idx] @ self.item_factors.T
        preds += self.user_means[u_idx]
        return np.clip(preds, 0.5, 5.0)

    def recommend_for_user(self, user_id, top_n: int = 10) -> pd.DataFrame:
        if user_id not in self.user_id_to_idx:
            raise ValueError(f"userId {user_id} not found in ratings.")
        preds = self.predict_ratings_for_user(user_id)
        u_idx = self.user_id_to_idx[user_id]
        already_rated = self._dense_known[u_idx] != 0

        candidates = np.argsort(preds)[::-1]
        candidates = [c for c in candidates if not already_rated[c]][:top_n]

        rows = []
        for i in candidates:
            mid = self.idx_to_movie_id[i]
            rows.append({
                "movieId": mid,
                "clean_title": self.movie_id_to_title[mid],
                "predicted_rating": round(float(preds[i]), 2),
            })
        return pd.DataFrame(rows)

    def evaluate_rmse(self, test_ratings: pd.DataFrame) -> float:
        """RMSE of predicted vs actual on a held-out set of ratings."""
        errs = []
        for row in test_ratings.itertuples():
            if row.userId not in self.user_id_to_idx or row.movieId not in self.movie_id_to_idx:
                continue
            u = self.user_id_to_idx[row.userId]
            m = self.movie_id_to_idx[row.movieId]
            pred = self.user_factors[u] @ self.item_factors[m] + self.user_means[u]
            pred = float(np.clip(pred, 0.5, 5.0))
            errs.append((pred - row.rating) ** 2)
        return float(np.sqrt(np.mean(errs))) if errs else float("nan")


if __name__ == "__main__":
    movies, ratings, links = load_clean_data()

    print("### Item-based CF: movies similar to 'Toy Story' (by rating patterns) ###")
    item_cf = ItemBasedCF(ratings, movies).fit()
    print(item_cf.recommend("Toy Story", top_n=5).to_string(index=False))

    print("\n### Matrix-factorization CF: top picks for userId=1 ###")
    mf_cf = MatrixFactorizationCF(ratings, movies, n_factors=30).fit()
    print(mf_cf.recommend_for_user(1, top_n=5).to_string(index=False))
