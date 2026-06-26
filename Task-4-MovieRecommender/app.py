import pandas as pd

movies = {
    "movie": [
        "Avengers", "Iron Man", "Thor",
        "Batman", "Superman", "Spiderman",
        "Inception", "Interstellar", "Titanic"
    ],
    "genre": [
        "action", "action", "action",
        "action", "action", "action",
        "sci-fi", "sci-fi", "romance"
    ]
}

df = pd.DataFrame(movies)

def recommend(movie_name):
    if movie_name not in df["movie"].values:
        return []

    genre = df[df["movie"] == movie_name]["genre"].values[0]
    recommendations = df[df["genre"] == genre]["movie"].tolist()
    recommendations.remove(movie_name)
    return recommendations

print("🎬 MOVIE RECOMMENDATION SYSTEM")

while True:
    user_input = input("\nEnter a movie name (or type 'exit'): ")

    if user_input.lower() == "exit":
        print("Goodbye 👋")
        break

    result = recommend(user_input)

    if len(result) == 0:
        print("Movie not found ❌")
    else:
        print("\nRecommended Movies:")
        for movie in result:
            print("👉", movie)
