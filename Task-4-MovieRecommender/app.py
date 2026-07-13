import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================================
# 1. SAMPLE DATA
# In a real project, load this from a CSV (e.g. MovieLens dataset).
# =====================================================================
MOVIES = pd.DataFrame({
    "movie_id": range(1, 13),
    "title": [
        "The Matrix", "Inception", "Interstellar", "The Notebook",
        "Titanic", "La La Land", "The Dark Knight", "Joker",
        "Toy Story", "Finding Nemo", "The Conjuring", "Get Out",
    ],
    "genres": [
        "Action Sci-Fi Thriller", "Action Sci-Fi Thriller Mystery",
        "Sci-Fi Drama Adventure", "Romance Drama",
        "Romance Drama", "Romance Musical Drama",
        "Action Crime Thriller Drama", "Crime Drama Thriller",
        "Animation Family Comedy", "Animation Family Comedy",
        "Horror Thriller", "Horror Mystery Thriller",
    ],
})

# user_id, movie_id, rating (1-5). Simulates a ratings database.
RATINGS = pd.DataFrame({
    "user_id": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 1, 2, 3],
    "movie_id": [1, 2, 7, 3, 4, 5, 1, 8, 7, 9, 10, 6, 11, 12, 7, 3, 8, 6],
    "rating":   [5, 4, 5, 5, 4, 4, 4, 5, 4, 5, 4, 3, 5, 4, 3, 3, 4, 2],
})


# =====================================================================
# 2. CONTENT-BASED FILTERING
# Recommends movies similar to one the user already likes, using
# TF-IDF vectors of genre text + cosine similarity between movies.
# =====================================================================
class ContentBasedRecommender:
    def __init__(self, movies_df):
        self.movies = movies_df.reset_index(drop=True)
        self._build_similarity_matrix()

    def _build_similarity_matrix(self):
        # TF-IDF converts each movie's genre string into a weighted
        # vector of terms. Movies sharing more/rarer genre terms end
        # up "closer" to each other in this vector space.
        tfidf = TfidfVectorizer(stop_words="english")
        tfidf_matrix = tfidf.fit_transform(self.movies["genres"])

        # Cosine similarity measures the angle between two movies'
        # vectors -- 1.0 means identical genre profile, 0 means no overlap.
        self.similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    def recommend(self, movie_title, top_n=5):
        """Return the top_n movies most similar to the given movie title."""
        matches = self.movies[self.movies["title"].str.lower() == movie_title.lower()]
        if matches.empty:
            return f"Movie '{movie_title}' not found in the dataset."

        idx = matches.index[0]
        similarity_scores = list(enumerate(self.similarity_matrix[idx]))

        # Sort by similarity, excluding the movie itself (always similarity=1.0)
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        similarity_scores = [s for s in similarity_scores if s[0] != idx][:top_n]

        recommendations = self.movies.iloc[[i for i, _ in similarity_scores]].copy()
        recommendations["similarity_score"] = [round(score, 3) for _, score in similarity_scores]
        return recommendations[["title", "genres", "similarity_score"]]


# =====================================================================
# 3. COLLABORATIVE FILTERING (User-Based)
# Recommends movies liked by OTHER users whose rating patterns are
# similar to the target user's -- "users like you also enjoyed..."
# =====================================================================
class CollaborativeRecommender:
    def __init__(self, ratings_df, movies_df):
        self.movies = movies_df
        self.ratings = ratings_df
        self._build_user_item_matrix()

    def _build_user_item_matrix(self):
        # Rows = users, Columns = movies, values = ratings (0 if unrated)
        self.user_item_matrix = self.ratings.pivot_table(
            index="user_id", columns="movie_id", values="rating"
        ).fillna(0)

        # Cosine similarity between USERS based on their rating vectors.
        # Users who rate similar movies similarly end up "close" together.
        self.user_similarity = cosine_similarity(self.user_item_matrix)
        self.user_similarity_df = pd.DataFrame(
            self.user_similarity,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index,
        )

    def recommend(self, user_id, top_n=5):
        """
        Recommend movies for user_id using weighted ratings from similar users:
        for each unrated movie, predicted_score = sum(similarity * rating) / sum(similarity)
        across all other users, then rank by that score.
        """
        if user_id not in self.user_item_matrix.index:
            return f"User {user_id} not found in the dataset."

        similar_users = self.user_similarity_df[user_id].drop(user_id)
        user_ratings = self.user_item_matrix.loc[user_id]
        unrated_movies = user_ratings[user_ratings == 0].index

        predicted_scores = {}
        for movie_id in unrated_movies:
            movie_ratings_by_others = self.user_item_matrix[movie_id]
            weighted_sum = (similar_users * movie_ratings_by_others).sum()
            similarity_sum = similar_users[movie_ratings_by_others > 0].sum()

            if similarity_sum > 0:
                predicted_scores[movie_id] = weighted_sum / similarity_sum

        if not predicted_scores:
            return "Not enough data to generate recommendations for this user."

        ranked = sorted(predicted_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        movie_ids = [m for m, _ in ranked]
        scores = [round(s, 2) for _, s in ranked]

        result = self.movies[self.movies["movie_id"].isin(movie_ids)].copy()
        result = result.set_index("movie_id").loc[movie_ids].reset_index()
        result["predicted_rating"] = scores
        return result[["title", "genres", "predicted_rating"]]

# =====================================================================
# 4. USER INPUT
# =====================================================================
if __name__ == "__main__":

    content_recommender = ContentBasedRecommender(MOVIES)
    collab_recommender = CollaborativeRecommender(RATINGS, MOVIES)

    print("=" * 60)
    print("MOVIE RECOMMENDATION SYSTEM")
    print("=" * 60)

    print("\nChoose Recommendation Type")
    print("1. Content-Based Recommendation")
    print("2. Collaborative Recommendation")

    choice = input("\nEnter your choice (1 or 2): ")

    if choice == "1":
        movie = input("\nEnter a movie name: ")
        print("\nRecommended Movies:")
        print(content_recommender.recommend(movie, top_n=5))

    elif choice == "2":
        try:
            user_id = int(input("\nEnter User ID (1-5): "))
            print("\nRecommended Movies:")
            print(collab_recommender.recommend(user_id, top_n=5))
        except ValueError:
            print("Please enter a valid User ID.")

    else:
        print("Invalid choice!")
