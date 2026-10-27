import numpy as np


class EASE:
    def fit(self, user_item_matrix, regularization=500):
        gram_matrix = (user_item_matrix.T @ user_item_matrix).toarray()
        gram_matrix = gram_matrix.astype(np.float64)
        gram_matrix.flat[:: gram_matrix.shape[0] + 1] += regularization

        inverse = np.linalg.inv(gram_matrix)
        weights = -inverse / np.diag(inverse)[None, :]
        np.fill_diagonal(weights, 0)
        self.weights = weights
        self.user_item_matrix = user_item_matrix.tocsr()
        return self

    def predict(self, user_index):
        scores = self.user_item_matrix[user_index] @ self.weights
        return np.asarray(scores).ravel()
