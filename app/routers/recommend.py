import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from app.schemas import ByIndexRequest, ByGenreRequest, RecommendResponse, MovieResult
from app.model import get_knn, get_features, get_df, get_titles, get_genres

router = APIRouter(tags=["Recommendations"])


# ── Helper: look up title by row index ────────────────────────────────────
def _get_title(df, titles, idx):
    try:
        mid   = df.iloc[idx]["movieId"]
        match = titles[titles["movieId"] == mid]["title"].values
        return match[0] if len(match) else f"Movie #{idx}"
    except Exception:
        return f"Movie #{idx}"


# ── Helper: build a result list from KNN output ────────────────────────────
def _build_results(distances, indices, df, titles, genre_cols,
                   skip_index=None, min_rating=None, year_range=None, n=10):
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if skip_index is not None and idx == skip_index:
            continue

        movie = df.iloc[idx]

        if min_rating and movie["average_rating"] < min_rating:
            continue

        if year_range:
            yr = movie.get("year", None)
            if pd.notna(yr) and not (year_range[0] <= yr <= year_range[1]):
                continue

        year_val = movie.get("year", None)
        results.append(MovieResult(
            rank           = len(results) + 1,
            title          = _get_title(df, titles, idx),
            genres         = [g for g in genre_cols if movie[g] == 1],
            year           = int(year_val) if pd.notna(year_val) else None,
            average_rating = round(float(movie["average_rating"]), 2),
            similarity     = round(1 - float(dist), 4),
        ))

        if len(results) >= n:
            break

    return results


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 1: List all valid genres
# ══════════════════════════════════════════════════════════════════════════════
@router.get("/genres", summary="List all available genres")
def list_genres():
    """Returns the list of genres you can use in the /recommend/by-genre endpoint."""
    return {"genres": get_genres()}


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 2: Get a movie by index (useful to explore the dataset)
# ══════════════════════════════════════════════════════════════════════════════
@router.get("/movies/{movie_index}", summary="Get movie info by index")
def get_movie(movie_index: int):
    """Returns details about a single movie by its dataset row index."""
    df     = get_df()
    titles = get_titles()
    genres = get_genres()

    if movie_index < 0 or movie_index >= len(df):
        raise HTTPException(status_code=404, detail=f"Index {movie_index} out of range (0–{len(df)-1})")

    movie    = df.iloc[movie_index]
    year_val = movie.get("year", None)

    return {
        "index":          movie_index,
        "title":          _get_title(df, titles, movie_index),
        "genres":         [g for g in genres if movie[g] == 1],
        "year":           int(year_val) if pd.notna(year_val) else None,
        "average_rating": round(float(movie["average_rating"]), 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 3: Recommend by movie index
# ══════════════════════════════════════════════════════════════════════════════
@router.post("/recommend/by-index", response_model=RecommendResponse,
             summary="Recommend movies similar to a given movie index")
def recommend_by_index(body: ByIndexRequest):
    """
    Finds the N most similar movies to the one at `movie_index`
    using the trained KNN model.
    """
    knn      = get_knn()
    features = get_features()
    df       = get_df()
    titles   = get_titles()
    genres   = get_genres()

    if body.movie_index < 0 or body.movie_index >= len(df):
        raise HTTPException(status_code=404,
                            detail=f"movie_index must be between 0 and {len(df)-1}")

    query_vec          = features[body.movie_index].reshape(1, -1)
    distances, indices = knn.kneighbors(query_vec, n_neighbors=body.n + 1)

    year_range = (body.year_min, body.year_max) if body.year_min and body.year_max else None

    results = _build_results(
        distances, indices, df, titles, genres,
        skip_index=body.movie_index,
        min_rating=body.min_rating,
        year_range=year_range,
        n=body.n
    )

    return RecommendResponse(count=len(results), results=results)


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 4: Recommend by genre preferences
# ══════════════════════════════════════════════════════════════════════════════
@router.post("/recommend/by-genre", response_model=RecommendResponse,
             summary="Recommend movies by preferred genres")
def recommend_by_genre(body: ByGenreRequest):
    """
    Builds a virtual 'ideal movie' vector from your preferred genres,
    then uses KNN to find the closest real movies.
    """
    knn      = get_knn()
    features = get_features()
    df       = get_df()
    titles   = get_titles()
    genres   = get_genres()

    # Validate genres
    invalid = [g for g in body.genres if g not in genres]
    if invalid:
        raise HTTPException(status_code=422,
                            detail=f"Unknown genres: {invalid}. Call GET /api/genres for valid options.")

    # Build query vector (genres = 1, year & rating stay at 0 = mean after scaling)
    query_vec = np.zeros(features.shape[1])
    for i, g in enumerate(genres):
        if g in body.genres:
            query_vec[i] = 1.0

    distances, indices = knn.kneighbors(
        query_vec.reshape(1, -1), n_neighbors=body.n * 2
    )

    results = _build_results(
        distances, indices, df, titles, genres,
        min_rating=body.min_rating,
        n=body.n
    )

    return RecommendResponse(count=len(results), results=results)