import numpy as np
from sklearn.preprocessing import normalize

from models.knn_utils import keep_top_neighbors


class ItemKNN:
    def fit(self, user_item_matrix, n_neighbors):
        self.user_item_matrix = user_item_matrix.tocsr()
        item_user_matrix = self.user_item_matrix.T.tocsr()
        normalized_matrix = normalize(item_user_matrix)
        similarity = normalized_matrix @ normalized_matrix.T
        similarity.setdiag(0)
        similarity.eliminate_zeros()
        self.similarity = keep_top_neighbors(similarity.tocsr(), n_neighbors)
        return self

    def predict(self, user_index):
        scores = self.user_item_matrix[user_index] @ self.similarity
        return np.asarray(scores.todense()).ravel()
