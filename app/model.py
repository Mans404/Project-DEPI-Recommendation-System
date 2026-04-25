import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# ── Paths to your saved files (adjust if needed) ───────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model_files" / "movie_knn_model.pkl"
FEAT_PATH  = BASE_DIR / "model_files" / "movie_knn_features.npy"
DATA_PATH  = BASE_DIR / "model_files" / "movie_knn_data.csv"
TITLE_PATH = BASE_DIR / "model_files" / "movie_knn_titles.csv"

GENRE_COLUMNS = [
    'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
    'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance',
    'Sci-Fi', 'Thriller', 'War', 'Western'
]

# ── Singleton state ────────────────────────────────────────────────────────
_resources = {}

def load_resources():
    """Called once at startup — loads everything into memory."""
    print("Loading model and data...")
    _resources["knn"]      = joblib.load(MODEL_PATH)
    _resources["features"] = np.load(FEAT_PATH)
    _resources["df"]       = pd.read_csv(DATA_PATH).reset_index(drop=True)
    _resources["titles"]   = pd.read_csv(TITLE_PATH)
    print(f"  Loaded {len(_resources['df'])} movies ✅")

def get_knn():
    return _resources["knn"]

def get_features():
    return _resources["features"]

def get_df():
    return _resources["df"]

def get_titles():
    return _resources["titles"]

def get_genres():
    return GENRE_COLUMNS