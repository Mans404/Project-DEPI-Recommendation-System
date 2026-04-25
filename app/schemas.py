from pydantic import BaseModel, Field
from typing import Optional

# ── Request bodies ─────────────────────────────────────────────────────────

class ByIndexRequest(BaseModel):
    movie_index: int = Field(..., description="Row index of the movie in the dataset")
    n:           int = Field(10,  ge=1, le=50, description="Number of recommendations")
    min_rating:  Optional[float] = Field(None, ge=0.5, le=5.0, description="Minimum average rating")
    year_min:    Optional[int]   = Field(None, description="Earliest release year")
    year_max:    Optional[int]   = Field(None, description="Latest release year")

    model_config = {
        "json_schema_extra": {
            "example": {
                "movie_index": 0,
                "n": 10,
                "min_rating": 3.5,
                "year_min": 1990,
                "year_max": 2020
            }
        }
    }


class ByGenreRequest(BaseModel):
    genres:     list[str] = Field(..., description="List of preferred genres")
    n:          int        = Field(10, ge=1, le=50, description="Number of recommendations")
    min_rating: float      = Field(3.0, ge=0.5, le=5.0, description="Minimum average rating")

    model_config = {
        "json_schema_extra": {
            "example": {
                "genres": ["Action", "Sci-Fi"],
                "n": 10,
                "min_rating": 3.5
            }
        }
    }


# ── Response bodies ────────────────────────────────────────────────────────

class MovieResult(BaseModel):
    rank:           int
    title:          str
    genres:         list[str]
    year:           Optional[int]
    average_rating: float
    similarity:     float


class RecommendResponse(BaseModel):
    count:   int
    results: list[MovieResult]