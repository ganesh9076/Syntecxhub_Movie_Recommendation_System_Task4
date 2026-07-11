# %%
"""
evaluate.py
------------
Evaluates and compares the content-based and collaborative-filtering
recommenders:

 1. Qualitative: run a handful of sample queries through each model and
    print/save the top-N results side by side, so you can visually judge
    whether the recommendations make sense.
 2. Quantitative (bonus, for the matrix-factorization CF only, since it's
    the only model that predicts a numeric rating): held-out RMSE on a
    train/test split of ratings.csv.

Run: python src/evaluate.py   (from the project root)
Writes a markdown report to outputs/evaluation_report.md
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_prep import load_clean_data
from content_based import ContentBasedRecommender
from collaborative import ItemBasedCF, MatrixFactorizationCF

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

SAMPLE_MOVIE_QUERIES = [
    "Toy Story",
    "The Matrix",
    "The Dark Knight",
    "Forrest Gump",
    "The Shawshank Redemption",
]
SAMPLE_USER_IDS = [1, 45, 234]


def train_test_split_ratings(ratings: pd.DataFrame, test_frac=0.2, seed=42):
    rng = np.random.RandomState(seed)
    # split per user so every user appears in both train & test where possible
    test_idx = []
    for uid, grp in ratings.groupby("userId"):
        n_test = max(1, int(len(grp) * test_frac))
        test_idx.extend(rng.choice(grp.index, size=n_test, replace=False))
    test = ratings.loc[test_idx]
    train = ratings.drop(index=test_idx)
    return train, test


def run_qualitative(content_model, item_cf, movies, report_lines):
    report_lines.append("## 1. Qualitative Evaluation — Sample Queries\n")
    report_lines.append(
        "For each seed movie, we show the top-5 results from the "
        "**content-based** model (similar genres/tags) and the "
        "**item-based collaborative filtering** model (similar rating patterns).\n"
    )

    for query in SAMPLE_MOVIE_QUERIES:
        print(f"\n{'='*70}\nSEED MOVIE: {query}\n{'='*70}")
        report_lines.append(f"### Seed movie: *{query}*\n")

        try:
            cb = content_model.recommend(query, top_n=5)
            print("\n-- Content-based (TF-IDF genres+tags, cosine similarity) --")
            print(cb[["clean_title", "genres", "similarity"]].to_string(index=False))
            report_lines.append("**Content-based (metadata similarity):**\n")
            report_lines.append(cb[["clean_title", "genres", "similarity"]].to_markdown(index=False) + "\n")
        except ValueError as e:
            print("Content-based error:", e)
            report_lines.append(f"_Content-based: {e}_\n")

        try:
            icf = item_cf.recommend(query, top_n=5)
            print("\n-- Item-based CF (rating pattern similarity) --")
            print(icf[["clean_title", "similarity"]].to_string(index=False))
            report_lines.append("**Item-based collaborative filtering (rating-pattern similarity):**\n")
            report_lines.append(icf[["clean_title", "similarity"]].to_markdown(index=False) + "\n")
        except ValueError as e:
            print("Item-based CF error:", e)
            report_lines.append(f"_Item-based CF: {e}_\n")


def run_user_recs(mf_model, ratings, movies, report_lines):
    report_lines.append("\n## 2. Personalized Recommendations (Matrix Factorization CF)\n")
    report_lines.append(
        "For a few sample users, we show 3 movies they rated highly (their taste) "
        "next to the model's top-5 predicted recommendations from movies they haven't seen.\n"
    )

    title_map = movies.set_index("movieId")["clean_title"]
    for uid in SAMPLE_USER_IDS:
        user_ratings = ratings[ratings["userId"] == uid].sort_values("rating", ascending=False)
        if user_ratings.empty:
            continue
        liked = user_ratings.head(3)["movieId"].map(title_map).tolist()
        recs = mf_model.recommend_for_user(uid, top_n=5)

        print(f"\n{'='*70}\nUSER {uid} — liked (top-rated): {liked}\n{'='*70}")
        print(recs.to_string(index=False))

        report_lines.append(f"### User {uid}\n")
        report_lines.append(f"- Rated highly: {', '.join(liked)}\n")
        report_lines.append("- Recommended:\n")
        report_lines.append(recs.to_markdown(index=False) + "\n")


def run_quantitative(ratings, movies, report_lines):
    report_lines.append("\n## 3. Quantitative Evaluation — Matrix Factorization RMSE\n")
    train, test = train_test_split_ratings(ratings, test_frac=0.2)
    print(f"\nTrain: {len(train)} ratings | Test: {len(test)} ratings")

    results = {}
    for k in (10, 30, 60):
        model = MatrixFactorizationCF(train, movies, n_factors=k).fit()
        rmse = model.evaluate_rmse(test)
        results[k] = rmse
        print(f"n_factors={k:>3}  ->  RMSE = {rmse:.4f}")

    report_lines.append(
        "Held-out RMSE (lower is better) of predicted vs actual star rating, "
        f"on an 80/20 per-user train/test split ({len(train)} train / {len(test)} test ratings):\n"
    )
    res_df = pd.DataFrame({"n_factors": list(results.keys()), "RMSE": list(results.values())})
    report_lines.append(res_df.to_markdown(index=False) + "\n")
    report_lines.append(
        "\n_For reference, always predicting the global mean rating (~3.5) typically "
        "gives an RMSE around 1.0-1.05 on this dataset, so this is our naive baseline._\n"
    )

    baseline_pred = train["rating"].mean()
    baseline_rmse = np.sqrt(((test["rating"] - baseline_pred) ** 2).mean())
    print(f"Baseline (predict global mean {baseline_pred:.2f}) -> RMSE = {baseline_rmse:.4f}")
    report_lines.append(f"- Baseline (predict global mean {baseline_pred:.2f}): RMSE = {baseline_rmse:.4f}\n")


def main():
    movies, ratings, links = load_clean_data()

    content_model = ContentBasedRecommender(movies).fit()
    item_cf = ItemBasedCF(ratings, movies).fit()
    mf_cf = MatrixFactorizationCF(ratings, movies, n_factors=30).fit()

    report_lines = ["# Movie Recommender — Evaluation Report\n"]

    run_qualitative(content_model, item_cf, movies, report_lines)
    run_user_recs(mf_cf, ratings, movies, report_lines)
    run_quantitative(ratings, movies, report_lines)

    report_path = os.path.join(OUT_DIR, "evaluation_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\nFull report written to: {report_path}")


if __name__ == "__main__":
    main()
