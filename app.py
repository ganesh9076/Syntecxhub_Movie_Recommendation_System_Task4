# %%
"""
app.py
------
Small Flask API exposing both recommenders.

Run:
    python app.py
Then visit http://127.0.0.1:5000/ for a simple HTML form, or hit the
JSON endpoints directly:

    GET /api/recommend/content?title=Toy%20Story&n=10
    GET /api/recommend/item_cf?title=Toy%20Story&n=10
    GET /api/recommend/user_cf?user_id=1&n=10
    GET /api/movies?q=matrix          (search the catalog for a title)
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from flask import Flask, request, jsonify, render_template
from src.data_prep import load_clean_data
from src.content_based import ContentBasedRecommender
from src.collaborative import ItemBasedCF, MatrixFactorizationCF

app = Flask(__name__)

print("Loading data and training models (this happens once at startup)...")
MOVIES, RATINGS, LINKS = load_clean_data()
CONTENT_MODEL = ContentBasedRecommender(MOVIES).fit()
ITEM_CF_MODEL = ItemBasedCF(RATINGS, MOVIES).fit()
USER_CF_MODEL = MatrixFactorizationCF(RATINGS, MOVIES, n_factors=30).fit()
print("Models ready.")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/movies")
def search_movies():
    q = request.args.get("q", "").strip().lower()
    if not q:
        return jsonify([])
    matches = MOVIES[MOVIES["clean_title"].str.lower().str.contains(q, na=False)]
    matches = matches.sort_values("year", ascending=False).head(15)
    return jsonify(matches[["movieId", "clean_title", "year", "genres"]].to_dict("records"))


@app.route("/api/recommend/content")
def recommend_content():
    title = request.args.get("title", "")
    n = int(request.args.get("n", 10))
    if not title:
        return jsonify({"error": "Provide a 'title' query parameter."}), 400
    try:
        recs = CONTENT_MODEL.recommend(title, top_n=n)
        return jsonify({
            "method": "content_based",
            "matched_title": recs["matched_query_title"].iloc[0] if len(recs) else title,
            "recommendations": recs.drop(columns=["matched_query_title"]).to_dict("records"),
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 404


@app.route("/api/recommend/item_cf")
def recommend_item_cf():
    title = request.args.get("title", "")
    n = int(request.args.get("n", 10))
    if not title:
        return jsonify({"error": "Provide a 'title' query parameter."}), 400
    try:
        recs = ITEM_CF_MODEL.recommend(title, top_n=n)
        return jsonify({
            "method": "item_based_collaborative_filtering",
            "matched_title": recs["matched_query_title"].iloc[0] if len(recs) else title,
            "recommendations": recs.drop(columns=["matched_query_title"]).to_dict("records"),
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 404


@app.route("/api/recommend/user_cf")
def recommend_user_cf():
    user_id = request.args.get("user_id", type=int)
    n = int(request.args.get("n", 10))
    if user_id is None:
        return jsonify({"error": "Provide a 'user_id' query parameter (integer)."}), 400
    try:
        recs = USER_CF_MODEL.recommend_for_user(user_id, top_n=n)
        return jsonify({
            "method": "matrix_factorization_collaborative_filtering",
            "user_id": user_id,
            "recommendations": recs.to_dict("records"),
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 404


if __name__ == "__main__":
    app.run(debug=True, port=5000)
