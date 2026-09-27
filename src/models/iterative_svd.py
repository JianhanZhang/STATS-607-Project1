import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import svds

class IterativeSVD:
    def __init__(self, k=10, num_epochs=10):
        # Initialize instance variables
        self.k = k
        self.num_epochs = num_epochs
        self.R = None
        self.global_mean = 0
        self.user_map = None
        self.movie_map = None
        self.user_means = []
        self.movie_means = []

    def fit(self, X, y):
        # Map user_id and movie_id to indices
        users = X['user_id'].unique()
        movies = X['movie_id'].unique()
        user_map = {user_id: idx for idx, user_id in enumerate(users)}
        movie_map = {movie_id: idx for idx, movie_id in enumerate(movies)}
        self.user_map = user_map
        self.movie_map = movie_map
        self.global_mean = np.mean(y)

        for u in users:
            self.user_means.append(np.mean(y[X['user_id'] == u]))
        for m in movies:
            self.movie_means.append(np.mean(y[X['movie_id'] == m]))
        self.user_means = np.array(self.user_means)
        self.movie_means = np.array(self.movie_means)

        # Create the sparse ratings matrix to save memory
        num_users = len(users)
        num_movies = len(movies)
        R = lil_matrix((num_users, num_movies))
        for (user, movie, rating) in zip(X['user_id'], X['movie_id'], y):
            R[user_map[user], movie_map[movie]] = rating - 0.5 * self.user_means[user_map[user]] - 0.5 * self.movie_means[movie_map[movie]]
        R = R.tocsr()

        # Main loop
        for epoch in range(self.num_epochs):
            # Perform SVD
            U, sigma, VT = svds(R, k=self.k)
            sigma = np.diag(sigma)

            # Reconstruct sparse matrix
            R_reconstructed = csr_matrix(U) @ csr_matrix(sigma) @ csr_matrix(VT)

            # Update the non-zero entries
            mask_non_zero = R != 0
            mask_zero = R == 0
            R = R.multiply(mask_non_zero) + R_reconstructed.multiply(mask_zero)
            self.R = R

            # Logging curent progress
            print(f"Epoch {epoch + 1} completed.")


    def predict(self, X):
        # Get the corresponding index for each user id and each movie_id
        user_indices = np.array([self.user_map.get(u, -1) for u in X['user_id']])
        item_indices = np.array([self.movie_map.get(i, -1) for i in X['movie_id']])

        # Initialize predictions
        predictions = np.full(len(X), self.global_mean)  # Set global mean as default prediction value
        valid_mask = (user_indices != -1) & (item_indices != -1)

        # Loop to predict each data point
        for idx in np.where(valid_mask)[0]:
            user_idx = user_indices[idx]
            item_idx = item_indices[idx]
            predictions[idx] = (
                self.R[user_idx, item_idx]
                + 0.5 * self.user_means[user_idx]
                + 0.5 * self.movie_means[item_idx]
            )

        return predictions


    '''def get_full_predictions(self):
        return self.R.toarray() + self.global_mean'''
