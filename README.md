# 🎬 Intelligent Movie Recommendation System

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.5+-orange?logo=scikit-learn)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-teal?logo=fastapi)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-green?logo=pandas)
![Status](https://img.shields.io/badge/Status-Deployed-brightgreen)

> **DEPI Final Project** — A complete end-to-end machine learning pipeline: from raw data cleaning and EDA, through KNN model training, to a live FastAPI backend with an interactive web UI — all built and deployed from scratch.

---

## Table of Contents

- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Full Pipeline](#-full-pipeline)
- [Installation & Usage](#-installation--usage)
- [API Endpoints](#-api-endpoints)
- [Methodology](#-methodology)
- [Model](#-model)
- [Results](#-results)
- [Live Demo](#-live-demo)
- [Contributors](#-contributors)
- [License](#-license)

---

## Problem Statement

With thousands of movies available across multiple streaming platforms, users often suffer from **"choice overload"**, making it difficult to decide what to watch next.

This project builds a complete recommendation system that:
- Cleans and preprocesses raw movie and ratings data
- Trains a KNN content-based model on genre, year, and rating features
- Serves recommendations through a REST API
- Displays results in a live interactive web interface

---

## Dataset

Based on the [MovieLens](https://grouplens.org/datasets/movielens/) format:

| File | Description |
|------|-------------|
| `Movies_Sample.csv` | Movie metadata: titles, genres, release years |
| `ratings_sample.csv` | User-item interactions and ratings |
| `merged_movies_and_ratings.csv` | Unified dataset after joining movies + ratings |
| `unique_movies_with_average_ratings.csv` | Aggregated average score per movie |

**Dataset Statistics:**
- 🎬 ~9,700 movies
- ⭐ ~100,000 ratings
- 👤 ~610 unique users
- 🏷️ 19 unique genres

---

## Project Structure

```
MOVIE-RECOMMENDATION-SYSTEM/
│
├── app/
│   ├── routers/
│   │   ├── __init__.py
│   │   └── recommend.py  
│   ├── __init__.py
│   ├── main.py            
│   ├── model.py             
│   └── schemas.py             ← Pydantic request/response models
│
├── data/                      ← Raw and processed CSV files
│
├── model_files/               ← Trained KNN model + feature matrix
│   ├── movie_knn_model.pkl
│   ├── movie_knn_features.npy
│   ├── movie_knn_data.csv
│   └── movie_knn_titles.csv
│
├── notebooks/
│   └── Movies_preprocessing.ipynb   ← Full pipeline: EDA → training → saving
│
├── static/
│   └── movies_ui.html         ← Frontend UI (served by FastAPI)
│
├── visuals/                   ← EDA charts and plots
│
├── README.md
└── requirements.txt
```

---

## Full Pipeline

This project covers every stage from raw data to live deployment:

```
Raw CSV Data
    ↓
1. Data Cleaning & Preprocessing    (notebooks/Movies_preprocessing.ipynb)
    ↓
2. Exploratory Data Analysis (EDA)
    ↓
3. Feature Engineering
    ↓
4. KNN Model Training & Saving
    ↓
5. FastAPI Backend (app/)
    ↓
6. Interactive Web UI (static/movies_ui.html)
    ↓
7. Deployment
```

---

## Installation & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/movie-recommendation-system.git
cd movie-recommendation-system
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate the Model Files

Open and run the notebook from start to finish:

```bash
jupyter notebook notebooks/Movies_preprocessing.ipynb
```

This will produce 4 files inside `model_files/`:
- `movie_knn_model.pkl`
- `movie_knn_features.npy`
- `movie_knn_data.csv`
- `movie_knn_titles.csv`

### 4. Run the Application

```bash
uvicorn app.main:app --reload
```

### 5. Open the UI

Visit **http://127.0.0.1:8000** in your browser.

The UI and API are served from the same application — no separate setup needed.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serves the web UI |
| `GET` | `/api/genres` | Returns all 19 available genres |
| `GET` | `/api/movies/{index}` | Get movie info by dataset index |
| `POST` | `/api/recommend/by-index` | Recommend movies similar to a given movie |
| `POST` | `/api/recommend/by-genre` | Recommend movies by preferred genres |

### Example — Recommend by genre:

```bash
curl -X POST http://127.0.0.1:8000/api/recommend/by-genre \
  -H "Content-Type: application/json" \
  -d '{"genres": ["Action", "Sci-Fi"], "n": 10, "min_rating": 3.5}'
```

### Example Response:

```json
{
  "count": 10,
  "results": [
    {
      "rank": 1,
      "title": "The Matrix",
      "genres": ["Action", "Sci-Fi"],
      "year": 1999,
      "average_rating": 4.2,
      "similarity": 0.9871
    }
  ]
}
```

---

## Methodology

### 1. Data Cleaning
- Removed movies with missing or empty genres
- Replaced `(no genres listed)` with `NaN` and dropped those rows
- Extracted release year from movie titles using regex
- Dropped duplicate entries

### 2. Exploratory Data Analysis
- Visualized top 10 most frequent genres
- Analyzed movie release frequency over time
- Computed average ratings per movie

### 3. Feature Engineering
- One-hot encoded 19 genre columns using `str.get_dummies('|')`
- Normalized `year` and `average_rating` using `StandardScaler`
- Built a final feature matrix of shape `(n_movies, 21)`

### 4. Model Training
- Trained a `NearestNeighbors` model with cosine distance
- Saved model, feature matrix, and data using `joblib` and `numpy`

### 5. API Development
- Built a REST API with **FastAPI**
- Model loads once at startup and stays in memory for fast responses
- Serves the frontend HTML directly — no separate web server needed

---

## Model

#### K-Nearest Neighbors (Content-Based)

| Parameter | Value |
|-----------|-------|
| Algorithm | Brute Force |
| Metric | Cosine Similarity |
| Neighbors | 11 (returns 10, skips self) |
| Features | 19 genres + normalized year + normalized rating |

**Why cosine similarity?**
Works well with sparse genre vectors — it measures the angle between two movies' feature profiles rather than raw distance, making it robust when most genre values are zero.

**Two recommendation modes:**
- **By movie index** — finds the N most similar movies to a given movie
- **By genre preference** — builds a virtual ideal-movie vector and finds the closest real matches

---

## Results

### EDA Insights
- Drama and Comedy dominate the dataset (40%+ of all movies)
- Clear upward trend in movie production from the 1990s, peaking in the 2000s–2010s

### Model Output — Sample Recommendations for Action + Sci-Fi:

| Rank | Title | Genres | Rating | Similarity |
|------|-------|--------|--------|------------|
| 1 | The Matrix (1999) | Action, Sci-Fi | ⭐ 4.2 | 98.7% |
| 2 | Inception (2010) | Action, Sci-Fi | ⭐ 4.1 | 97.3% |
| 3 | Interstellar (2014) | Adventure, Sci-Fi | ⭐ 4.0 | 95.1% |

---

## Live Demo

The application is fully deployed and accessible at:

> 🔗 **[-deployment-url]**  | Note: I do not have Hosting for now, but I will buy it soon.

Alternatively, run it locally following the [Installation](#-installation--usage) steps above.

---

## Contributors

| Name | Role |
|------|------|
| Yousef Omran | Data Preprocessing & EDA |
| Mohamed Mansour | Model Development & API |
| Ali Yahia | Feature Engineering & Deployment |

---

> *Built as a final project for the **Digital Egypt Pioneers Initiative (DEPI)** — AI & Data Science Track.*
