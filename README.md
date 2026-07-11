# Movie Recommendation System

Internship Project 4 — content-based and collaborative-filtering movie
recommenders built on the MovieLens `ml-latest-small` dataset.

## Dataset

**MovieLens ml-latest-small** ([GroupLens](https://grouplens.org/datasets/movielens/)):
- `movies.csv` — 9,742 movies (title, year, pipe-separated genres)
- `ratings.csv` — 100,836 ratings from 610 users (0.5–5.0 stars)
- `tags.csv` — 3,683 free-text tags users applied to movies
- `links.csv` — IMDB / TMDB id cross-reference

Already downloaded into `data/`, so the project runs immediately with no
extra setup.

## Approach

Two independent recommenders are built and compared, exactly as the two
options in the brief:

| | Content-based | Collaborative Filtering |
|---|---|---|
| **Signal used** | Movie metadata only (genres + aggregated user tags) | User rating patterns only (no metadata) |
| **Technique** | TF-IDF vectorization of a "metadata soup" per movie + cosine similarity | (a) Item-based: cosine similarity between movies' rating vectors. (b) Matrix factorization: Truncated SVD on the mean-centered user-item matrix, predicting ratings for unseen movies |
| **Good for** | New/rarely-rated movies, explainable "similar genre/vibe" recs | Capturing patterns text can't ("people who liked X also liked Y") |
| **Cold-start weakness** | Ignores what users actually rated | Can't recommend a movie nobody has rated yet |

Both are implemented so you can see the trade-offs side by side in the
evaluation report.

## Project structure

```
movie_project/
├── data/                      # raw MovieLens CSVs
├── src/
│   ├── data_prep.py           # loading, cleaning, "metadata soup" builder
│   ├── eda.py                 # EDA: stats + plots -> outputs/figures/
│   ├── content_based.py        # TF-IDF + cosine similarity recommender
│   ├── collaborative.py       # item-based CF + matrix-factorization CF
│   └── evaluate.py            # qualitative + quantitative evaluation
├── templates/
│   └── index.html             # tiny UI for the Flask app
├── outputs/
│   ├── figures/                # EDA plots (generated on run)
│   └── evaluation_report.md    # generated evaluation report
├── app.py                      # Flask API + demo UI
├── run_pipeline.py             # one-shot: EDA + build + evaluate
└── requirements.txt
```

Every file in `src/` has `# %%` cell markers, so in VS Code (with the
Python/Jupyter extension) you can open any file and run it cell-by-cell in
the Interactive Window, in addition to running them as plain scripts.

## How to run

```bash
cd movie_project
pip install -r requirements.txt

# Run everything (EDA + build models + evaluate) in one go:
python run_pipeline.py

# Or run pieces individually:
python src/eda.py
python src/content_based.py
python src/collaborative.py
python src/evaluate.py

# Launch the Flask demo (http://127.0.0.1:5000):
python app.py
```

## What EDA / cleaning covers (`src/eda.py`, `src/data_prep.py`)

- Missing values and duplicate checks on all four raw files (dataset turns
  out to be already clean — no nulls, no duplicate IDs)
- Extracting release year from titles, and fixing MovieLens's
  `"Matrix, The (1999)"` → `"The Matrix (1999)"` inversion
- Rating distribution, ratings-per-user / ratings-per-movie (long-tail
  sparsity: ~98.3% of the user-item matrix is empty)
- Genre frequency breakdown
- Movies released per year
- Bayesian-adjusted "best movies" (average rating weighted by number of
  ratings, so a 5.0 from 2 people doesn't outrank Shawshank Redemption)

All plots are saved as PNGs to `outputs/figures/`.

## Evaluation (`src/evaluate.py`)

1. **Qualitative** — 5 seed movies (Toy Story, The Matrix, The Dark
   Knight, Forrest Gump, The Shawshank Redemption) run through both the
   content-based and item-based-CF models, so you can visually compare
   "genre-similar" vs "co-rated" recommendations for the same movie.
2. **Personalized** — for 3 sample users, shows what they rated highly
   next to the matrix-factorization model's top-5 predicted picks from
   movies they haven't seen yet.
3. **Quantitative** — held-out RMSE for the matrix-factorization model
   across a few different numbers of latent factors, compared against a
   "always predict the global mean" baseline (the model beats baseline
   RMSE ~1.05 down to ~0.92).

Full results get written to `outputs/evaluation_report.md` on every run.

## Flask API (`app.py`)

- `GET /` — small HTML demo page
- `GET /api/movies?q=matrix` — search the catalog for a title
- `GET /api/recommend/content?title=Toy%20Story&n=10` — content-based
- `GET /api/recommend/item_cf?title=Toy%20Story&n=10` — item-based CF
- `GET /api/recommend/user_cf?user_id=1&n=10` — personalized (matrix
  factorization)

## Notes / possible extensions

- Titles are matched fuzzily (case-insensitive + partial match +
  closest-match fallback), so `"matrix"` or `"drak knight"` still resolve.
- The metadata soup weights genres 3x more than tags, since tags are
  sparse (only ~1,572 of 9,742 movies have any) — this was a deliberate
  design choice, tunable in `data_prep.build_metadata_soup`.
- A natural next step would be a **hybrid** model (blend content-based
  and CF scores), or pulling in TMDB overviews/keywords via the `links.csv`
  TMDB ids for richer text features than genres+tags alone.

## Author

**Ganesh Palav**  
B.Tech Computer Science & Engineering
