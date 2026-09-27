import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from scipy.sparse import csr_matrix

class BRISMF(BaseEstimator):
    def __init__(self, K=5, lr_pb=0.0003, lr_p=0.0003, lr_qb=0.0003, lr_q=0.0003,
                 p_pb=0.01, p_p=0.01, p_qb=0.01, p_q=0.01, num_to_stop=2, epochs=10):
        # Initialize instance variables
        self.N = 0
        self.M = 0
        self.K = K
        self.lr_pb = lr_pb
        self.lr_p = lr_p
        self.lr_qb = lr_qb
        self.lr_q = lr_q
        self.p_pb = p_pb
        self.p_p = p_p
        self.p_qb = p_qb
        self.p_q = p_q
        self.num_to_stop = num_to_stop
        self.epochs = epochs
        self.P = None
        self.Q = None
        self.user_map = None
        self.movie_map = None
        self.global_mean = 0
        self.user_means = []
        self.movie_means = []


    def fit(self, X, y):
        # Map user_id and movie_id to indices
        self.global_mean = np.mean(y)
        users, user_map = np.unique(X['user_id'], return_inverse=True)
        movies, movie_map = np.unique(X['movie_id'], return_inverse=True)
        self.user_map = {u: i for i, u in enumerate(users)}
        self.movie_map = {i: j for j, i in enumerate(movies)}
        self.N = len(users)
        self.M = len(movies)
        self.P = self.generate_P0(self.N, self.K)
        self.Q = self.generate_Q0(self.K, self.M)

        # Compute mean rating for each user and each movie, respectively
        for u in users:
            self.user_means.append(np.mean(y[X['user_id'] == u]))
        for m in movies:
            self.movie_means.append(np.mean(y[X['movie_id'] == m]))
        self.user_means = np.array(self.user_means)
        self.movie_means = np.array(self.movie_means)

        # Standardize the training ratings by subtracting the average of their respective user mean rating and movie mean rating from them
        y_user_mean = np.zeros(X.shape[0])
        y_movie_mean = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            user_mean = self.user_means[self.user_map[X.iloc[i, 1]]]
            movie_mean = self.movie_means[self.movie_map[X.iloc[i, 0]]]
            y_user_mean[i] = user_mean
            y_movie_mean[i] = movie_mean
        y_standardized = y - 0.5 * y_user_mean - 0.5 * y_movie_mean

        # Create a sparse matrix of the standardized training ratings to save memory
        ratings = csr_matrix((y_standardized, (user_map, movie_map)), shape=(self.N, self.M))

        # Main loop
        num_epochs = 0
        best_rmse = np.inf
        while (num_epochs < self.epochs):
            for u, i, r in zip(*ratings.nonzero(), ratings.data):
                # Compute error
                pred = np.dot(self.P[u], self.Q[:, i])
                e = r - pred

                # Gradient updates
                if (u == 1):
                    self.P[u] += self.lr_pb * (e * self.Q[:, i] - self.p_pb * self.P[u])
                elif (u > 1):
                    self.P[u] += self.lr_p * (e * self.Q[:, i] - self.p_p * self.P[u])
                if (i == 0):
                    self.Q[:, i] += self.lr_qb * (e * self.P[u] - self.p_qb * self.Q[:, i])
                elif (i > 1):
                    self.Q[:, i] += self.lr_q * (e * self.P[u] - self.p_q * self.Q[:, i])
            num_epochs += 1

        # Logging
        print("Training finished with", num_epochs, "epochs and a final RMSE of", best_rmse)
        return self


    # Returns a list
    def predict(self, X):
        output = []

        # Get the corresponding index for each user id and each movie_id
        user_indices = X['user_id'].unique()
        movie_indices = X['movie_id'].unique()

        # Loop through the data to predict each data point
        for c in range(0, X.shape[0]):
            u_raw = X.iloc[c, 1] # Get the current user_id
            i_raw = X.iloc[c, 0] # Get the current movie_id

            # If the both current user and movie exist in training data, the prediction will be the dot product of the u-th row in P
            # and the i-th column in Q, plus the average of the user mean rating and the movie mean rating
            if (u_raw in self.user_map and i_raw in self.movie_map):
                u = self.user_map[X.iloc[c, 1]] # Get the index of the user
                i = self.movie_map[X.iloc[c, 0]] # Get the index of the movie
                r_hat = np.dot(self.P[u], self.Q[:,i])
                output.append(r_hat + 0.5 * self.movie_means[i] + 0.5 * self.user_means[u])
            # If the curent user, but not the curent movie, exists in training data, we use the user mean rating as the prediction
            elif (u_raw in self.user_map):
                u = self.user_map[X.iloc[c, 1]]
                output.append(self.user_means[u])
            # If the current movie, but not the current user, exists in training darta, we use the movie mean rating as the prediction
            elif (i_raw in self.movie_map):
                i = self.movie_map[X.iloc[c, 0]]
                output.append(self.movie_means[i])
            # If neither the current user nor the current movie exists in training data, we use the global mean rating as the prediction
            else:
                output.append(self.global_mean)

        return np.array(output)


    def get_P(self):
        return self.P


    def get_Q(self):
        return self.Q


    def generate_P0(self, N, K):
        output = 0.000001 * np.random.rand(N, K)
        output[:,0] = 1
        return output


    def generate_Q0(self, K, M):
        output = 0.000001 * np.random.rand(K, M)
        output[1,:] = 1
        return output


    def compute_rmse(self, ratings):
        rows, cols = ratings.nonzero()
        predictions = np.sum(self.P[rows] * self.Q[:, cols].T, axis=1)
        errors = ratings.data - predictions
        return np.sqrt(np.mean(errors ** 2))


    # Define get_params method to return the model's hyperparameters
    def get_params(self, deep=True):
        return {
            'K': self.K,
            'lr_pb': self.lr_pb,
            'lr_p': self.lr_p,
            'lr_qb': self.lr_qb,
            'lr_q': self.lr_q,
            'p_pb': self.p_pb,
            'p_p': self.p_p,
            'p_qb': self.p_qb,
            'p_q': self.p_q,
            'num_to_stop': self.num_to_stop,
        }


    # Define set_params method to set hyperparameters
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self