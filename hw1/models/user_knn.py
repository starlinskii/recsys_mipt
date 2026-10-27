import numpy as np
from sklearn.preprocessing import normalize

from models.knn_utils import keep_top_neighbors


class UserKNN:
    def fit(self, user_item_matrix, n_neighbors):
        self.user_item_matrix = user_item_matrix.tocsr()
        normalized_matrix = normalize(self.user_item_matrix)
        similarity = normalized_matrix @ normalized_matrix.T
        similarity.setdiag(0)
        similarity.eliminate_zeros()
        self.similarity = keep_top_neighbors(similarity.tocsr(), n_neighbors)
        return self

    def predict(self, user_index):
        scores = self.similarity[user_index] @ self.user_item_matrix
        return np.asarray(scores.todense()).ravel()
