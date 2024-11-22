import numpy as np
import matplotlib.pyplot as plt
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

# Loading movies
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

# Evaluation Metrics
def calculate_mae(predicted, actual):
    mask = actual > 0  # Only compare non-zero ratings
    return np.mean(np.abs(predicted[mask] - actual[mask]))

def calculate_rmse(predicted, actual):
    mask = actual > 0  # Only compare non-zero ratings
    return np.sqrt(np.mean((predicted[mask] - actual[mask]) ** 2))

# Calculate MAE, RMSE, and Number of Ratings for Each User
user_mae = []
user_rmse = []
num_ratings = []

for user_id in range(1, num_users + 1):
    user_index = user_id - 1
    actual_ratings = rating_matrix[user_index]
    predicted_ratings_user = predicted_ratings[user_index]
    mae = calculate_mae(predicted_ratings_user, actual_ratings)
    rmse = calculate_rmse(predicted_ratings_user, actual_ratings)
    count = np.sum(actual_ratings > 0)  # Count non-zero ratings
    user_mae.append(mae)
    user_rmse.append(rmse)
    num_ratings.append(count)

# Plotting Separate Graphs
plt.figure(figsize=(12, 5))

# MAE Plot
plt.subplot(1, 2, 1)
plt.scatter(num_ratings, user_mae, color="blue", alpha=0.7, s=10)  # Reduced dot size with `s=10`
plt.xlabel("# of True Ratings per User")
plt.ylabel("MAE")
plt.title("MAE vs. Number of True Ratings")
plt.grid(alpha=0.3)

# RMSE Plot
plt.subplot(1, 2, 2)
plt.scatter(num_ratings, user_rmse, color="orange", alpha=0.7, s=10)  # Reduced dot size with `s=10`
plt.xlabel("# of True Ratings per User")
plt.ylabel("RMSE")
plt.title("RMSE vs. Number of True Ratings")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
