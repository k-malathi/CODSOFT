Task 4: Recommendation System

## CODSOFT Artificial Intelligence Internship

### 📌 Overview
This project implements a **movie recommendation system** using two
classic, complementary techniques:

1. **Content-Based Filtering** — recommends movies *similar to a movie
   you already like*, based on movie attributes (genres).
2. **Collaborative Filtering (User-Based)** — recommends movies *liked
   by other users with similar taste*, based on a user-item ratings matrix.

No external dataset download is required — a small built-in sample
dataset (12 movies, 5 users) demonstrates both approaches end-to-end.
The code is modular, so a real dataset (e.g. MovieLens) can be swapped
in with minimal changes.

---

### 🧠 How Each Approach Works

#### 1. Content-Based Filtering (`ContentBasedRecommender`)
- Each movie's genre string (e.g. `"Action Sci-Fi Thriller"`) is
  converted into a numerical vector using **TF-IDF** (Term
  Frequency–Inverse Document Frequency), which weights genre terms by
  how distinctive they are.
- **Cosine similarity** is computed between every pair of movie
  vectors — this measures the angle between two vectors, giving a
  score from 0 (no overlap) to 1 (identical genre profile).
- To recommend, the system finds movies with the highest similarity
  score to the movie the user specifies.

> Example: liking *"The Matrix"* (Action/Sci-Fi/Thriller) recommends
> *"Inception"* (Action/Sci-Fi/Thriller/Mystery) with high similarity,
> since they share the most genre overlap.

#### 2. Collaborative Filtering (`CollaborativeRecommender`)
- Builds a **user-item matrix**: rows = users, columns = movies,
  values = ratings (0 if unrated).
- Computes **cosine similarity between users** based on their rating
  patterns — users who rate the same movies similarly are considered
  "close" in taste.
- For each movie the target user hasn't rated, predicts a score as a
  **similarity-weighted average** of how similar users rated it:
  ```
  predicted_rating(movie) = Σ(similarity(user, other) × rating(other, movie))
                              ─────────────────────────────────────────────
                              Σ similarity(user, other)  [for others who rated it]
  ```
- Movies are ranked by this predicted rating, and the top N are recommended.

> Example: if User 2 rates movies similarly to Users who loved *"The
> Dark Knight"*, that movie gets recommended to User 2 even though
> they never rated it.

---

### ▶️ How to Run

**Requirements:**
```bash
pip install -r requirements_recsys.txt
```

**Run the demo:**
```bash
python recommendation_system.py
```

This runs both recommenders on the built-in sample dataset and prints:
- Content-based recommendations for 3 sample movies
- Collaborative-filtering recommendations for 3 sample users

**Use it in your own code:**
```python
from recommendation_system import ContentBasedRecommender, CollaborativeRecommender, MOVIES, RATINGS

# Content-based: "movies like this one"
content_rec = ContentBasedRecommender(MOVIES)
print(content_rec.recommend("Inception", top_n=5))

# Collaborative: "recommendations for this user"
collab_rec = CollaborativeRecommender(RATINGS, MOVIES)
print(collab_rec.recommend(user_id=3, top_n=5))
```

---

### 📂 Files
```
recommendation_system.py     # Both recommender implementations + demo
requirements_recsys.txt       # Python dependencies
README_recsys.md              # Project documentation (this file)
```

---

### 🧪 Example Output
```
Because you liked 'The Matrix', you might also enjoy:
          title                          genres  similarity_score
     Inception  Action Sci-Fi Thriller Mystery             0.857
  Interstellar          Sci-Fi Drama Adventure             0.512

Top recommendations for User 1:
          title                genres  predicted_rating
 The Conjuring       Horror Thriller              5.00
         Joker  Crime Drama Thriller              4.74
```

---

### 🔄 Using a Real Dataset (e.g. MovieLens)
Replace the built-in `MOVIES` and `RATINGS` DataFrames with data
loaded from CSV files:
```python
MOVIES = pd.read_csv("movies.csv")     # columns: movie_id, title, genres
RATINGS = pd.read_csv("ratings.csv")   # columns: user_id, movie_id, rating
```
The [MovieLens 100K/1M datasets](https://grouplens.org/datasets/movielens/)
follow this exact structure and work as a drop-in replacement.

---

### 🚀 Possible Extensions
- Combine both approaches into a **hybrid recommender** (weighted average of both scores)
- Add **matrix factorization** (SVD) for more scalable collaborative filtering
- Use **item-based** collaborative filtering instead of user-based
- Incorporate additional content features (plot summary, cast, director) using NLP embeddings
- Build a simple web UI (Streamlit) for interactive recommendations

---

### 🏷️ Internship Info
This task was completed as part of the **CODSOFT Artificial Intelligence
Internship**. Repository maintained under: `CODSOFT`

#codsoft #internship #artificialintelligence #recommendationsystem
