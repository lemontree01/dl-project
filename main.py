import numpy as np
from scipy.sparse.linalg import svds

# PATH
data_path = "u.data"
movies_path = "u.item"

# Loading users
with open(data_path, "r") as f:
    data = f.readlines()

user_ids = []
movie_ids = []
ratings = []

for line in data:
    user_id, movie_id, rating, _ = map(int, line.split("\t"))
    user_ids.append(user_id)
    movie_ids.append(movie_id)
    ratings.append(rating)

# Loading moview
movie_names = {}
with open(movies_path, "r", encoding="latin-1") as f:
    for line in f:
        parts = line.split("|")
        movie_id = int(parts[0])
        movie_name = parts[1]
        movie_names[movie_id] = movie_name

num_users = max(user_ids)
num_movies = max(movie_ids)

rating_matrix = np.zeros((num_users, num_movies), dtype=np.float32)
for user_id, movie_id, rating in zip(user_ids, movie_ids, ratings):
    rating_matrix[user_id - 1, movie_id - 1] = rating

# Normalize
avg_user = np.true_divide(rating_matrix.sum(axis=1), (rating_matrix != 0).sum(axis=1))
avg_user = np.nan_to_num(avg_user)
rating_matrix_normalized = rating_matrix - avg_user[:, np.newaxis]
rating_matrix_normalized[rating_matrix == 0] = 0

# SVD
U, sigma, Vt = svds(rating_matrix_normalized, k=50)
sigma = np.diag(sigma)

predicted_ratings = np.dot(np.dot(U, sigma), Vt) + avg_user[:, np.newaxis]

# Recommendation
def recommend_movies(user_id, num_recommendations=5):
    user_index = user_id - 1
    user_predictions = predicted_ratings[user_index]

    # Get movies the user has already rated
    rated_movie_indices = np.where(rating_matrix[user_index] > 0)[0]
    print(f"Movies rated by user {user_id}:")
    for movie_index in rated_movie_indices:
        movie_id = movie_index + 1
        print(f"Movie {movie_names[movie_id]} with id {movie_id} has true rating: {rating_matrix[user_index, movie_index]}")
        
    recommendations = []
    for movie_index, score in enumerate(user_predictions):
        if movie_index not in rated_movie_indices:
            recommendations.append((movie_index + 1, score))

    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
    return recommendations[:num_recommendations]

user_id = 2
recommended_movies = recommend_movies(user_id)
print("\nRecommendations for user {}:".format(user_id))
for movie_id, score in recommended_movies:
    print(f"Movie {movie_names[movie_id]} with id {movie_id} has prediction {score:.2f}")
