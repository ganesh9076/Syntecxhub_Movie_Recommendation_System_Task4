# %%
"""
run_pipeline.py
----------------
One-shot entry point that reproduces the whole project end-to-end:
  1. Loads & cleans the data
  2. Runs EDA (prints stats, saves plots to outputs/figures/)
  3. Fits the content-based and collaborative-filtering models
  4. Runs the qualitative + quantitative evaluation
     (prints to console, saves outputs/evaluation_report.md)

Run from the project root:
    python run_pipeline.py

In VS Code, you can also open any file in src/ and run it cell-by-cell
using the '# %%' markers (Jupyter/Interactive Window extension).
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from src.eda import run_eda
from  src.evaluate import main as run_evaluation


def main():
    print("\n" + "#" * 70)
    print("# STEP 1/2 — EXPLORATORY DATA ANALYSIS")
    print("#" * 70)
    run_eda()

    print("\n" + "#" * 70)
    print("# STEP 2/2 — BUILD MODELS + EVALUATE")
    print("#" * 70)
    run_evaluation()

    print("\nDone. Check the outputs/ folder for figures and the evaluation report.")
    print("Run `python app.py` to launch the Flask demo at http://127.0.0.1:5000")


if __name__ == "__main__":
    main()
