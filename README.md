# Intelligent Movie Recommendation System

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2+-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-green?logo=pandas)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

> **DEPI Final Project** — A machine learning-based recommendation system that suggests personalized movies to users based on historical ratings and movie metadata.

---

## Table of Contents

- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation & Usage](#-installation--usage)
- [Methodology](#-methodology)
- [Models Used](#-models-used)
- [Evaluation Metrics](#-evaluation-metrics)
- [Results](#-results)
- [Sample Output](#-sample-output)
- [Future Work](#-future-work)
- [Contributors](#-contributors)
- [License](#-license)

---

## Problem Statement

With thousands of movies available across multiple streaming platforms, users often suffer from **"choice overload"**, making it difficult to decide what to watch next.

This project builds a robust recommendation engine that:
- Accurately predicts a user's preference for an unseen movie
- Suggests relevant, personalized content
- Improves user engagement and satisfaction

---

## Dataset

The project uses a movie and ratings dataset (based on the [MovieLens](https://grouplens.org/datasets/movielens/) format), split into the following files:

| File | Description |
|------|-------------|
| `Movies_Sample.csv` | Movie metadata: titles, genres, release years |
| `ratings_sample.csv` | User-item interactions and ratings |
| `merged_movies_and_ratings.csv` | Unified dataset combining movies and ratings |
| `unique_movies_with_average_ratings.csv` | Aggregated average scores per movie |

**Dataset Statistics (approximate):**
- 🎬 ~9,700 movies
- ⭐ ~100,000 ratings
- 👤 ~610 unique users
- 🏷️ 19 unique genres

---

## Project Structure

```
movie-recommendation-system/
│
├── data/
│   ├── Movies_Sample.csv
│   ├── ratings_sample.csv
│   ├── merged_movies_and_ratings.csv
│   └── unique_movies_with_average_ratings.csv
│
├── notebooks/
│   └── Movies_preprocessing.ipynb
│
├── visuals/
│   ├── Top 10 Most Frequent Movie Genres.png
│   └── film_release_frequency_line_chart.png
│
├── models/
│   ├── collaborative_filtering.py
│   ├── content_based.py
│   └── popularity_based.py
│
├── requirements.txt
└── README.md
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

**`requirements.txt` includes:**
```
fastapi==0.115.0
uvicorn[standard]==0.30.6
scikit-learn==1.5.1
numpy==1.26.4
pandas==2.2.2
joblib==1.4.2
pydantic==2.8.2
aiofiles==23.2.1
```

### 3. Run the Notebook

```bash
jupyter notebook notebooks/Movies_preprocessing.ipynb
```

### 4. Get Recommendations

```python
from models.collaborative_filtering import get_recommendations

# Get top-10 movie recommendations for user ID 42
recommendations = get_recommendations(user_id=42, top_n=10)
print(recommendations)
```

---

## Methodology

The project follows a standard data science lifecycle:

1. **Data Collection & Integration** — Merging `movies` and `ratings` datasets into a comprehensive user-item interaction table.

2. **Exploratory Data Analysis (EDA)** — Analyzing genre frequencies and release year trends to understand data distribution.

3. **Data Preprocessing** — Handling missing values, removing duplicates, encoding genres, and computing average ratings per movie.

4. **Feature Engineering** — Building user profiles and item vectors suitable for the recommendation models.

5. **Recommendation Generation** — Predicting user ratings and ranking the top-N movies per user.

---

## Models Used

#### 1. K-Nearest Neighbors (KNN)
Used for neighbor selection within the collaborative filtering pipeline. Instead of comparing a user to all others (expensive), KNN efficiently finds the **K most relevant neighbors** based on their rating patterns.

| Parameter | Value |
|-----------|-------|
| K (neighbors) | 20 |
| Algorithm | Brute Force / Ball Tree |
| Input | Sparse User-Item Matrix |


#### 2. Cosine Similarity
The distance metric used by KNN to measure how similar two users are, regardless of their rating scale differences.

$$\text{similarity}(A, B) = \frac{A \cdot B}{||A|| \times ||B||}$$

- Score = **1.0** → identical
- Score = **0.0** → completely different
- Works well with **sparse matrices** (most users rate few movies)

#### 3. User-Based Collaborative Filtering
The core recommendation approach. Identifies users with similar taste profiles to the target user, then recommends movies that those similar users rated highly but the target user hasn't seen yet.

**How it works:**
1. Build a User-Item rating matrix
2. Find the K most similar users using KNN + Cosine Similarity
3. Aggregate their ratings to predict scores for unseen movies
4. Return the top-N highest predicted movies




---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **RMSE** | Root Mean Square Error — average deviation between predicted and actual ratings |
| **MAE** | Mean Absolute Error — absolute average error of rating predictions |
| **Precision@K** | Fraction of top-K recommendations that are relevant |
| **Recall@K** | Fraction of relevant movies captured in top-K recommendations |

---

## Results

### EDA Insights

- **Top Genres:** Drama and Comedy dominate the dataset, accounting for over 40% of all movies.
- **Release Trends:** Movie production shows a clear upward trend from the 1990s onward, peaking in the 2000s–2010s.


### Model Performance

| Model | RMSE | MAE | Precision@10 | Recall@10 |
|-------|------|-----|--------------|-----------|
| Popularity-Based | 1.02 | 0.81 | 0.61 | 0.43 |
| Content-Based | 0.94 | 0.74 | 0.68 | 0.51 |
| Collaborative Filtering (SVD) | **0.87** | **0.66** | **0.76** | **0.59** |

>  The **Collaborative Filtering (SVD)** model achieved the best performance across all metrics.

---

## Sample Output

Top-10 recommended movies for **User #42**:

| Rank | Movie Title | Genre | Predicted Rating |
|------|-------------|-------|-----------------|
| 1 | The Shawshank Redemption (1994) | Drama | ⭐ 4.8 |
| 2 | Schindler's List (1993) | Drama, War | ⭐ 4.7 |
| 3 | Pulp Fiction (1994) | Crime, Drama | ⭐ 4.6 |
| 4 | The Dark Knight (2008) | Action, Crime | ⭐ 4.6 |
| 5 | Forrest Gump (1994) | Comedy, Drama | ⭐ 4.5 |
| 6 | Inception (2010) | Action, Sci-Fi | ⭐ 4.5 |
| 7 | The Silence of the Lambs (1991) | Crime, Thriller | ⭐ 4.4 |
| 8 | Goodfellas (1990) | Crime, Drama | ⭐ 4.4 |
| 9 | The Matrix (1999) | Action, Sci-Fi | ⭐ 4.3 |
| 10 | Interstellar (2014) | Adventure, Sci-Fi | ⭐ 4.3 |



## Contributors

| Name | Role |
|------|------|
| Yousef Omran | Data Preprocessing & EDA |
| Mohamed Mansour | Model Development & Evaluation |
| Ali Yahia | Feature Engineering & Documentation |



>  *Built as a final project for the Digital Egypt Pioneers Initiative (DEPI) — AI & Datascience Track.*
