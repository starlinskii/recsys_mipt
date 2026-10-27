from .custom_ials import CustomTargetALS


class ALS:
    def fit(self, user_item_matrix, factors=32, regularization=0.1, iterations=15):
        target_matrix = user_item_matrix.copy().tocsr()
        target_matrix.data[:] = 1
        self.model = CustomTargetALS().fit(
            user_item_matrix,
            target_matrix,
            factors=factors,
            iterations=iterations,
            alpha=40,
            regularization=regularization,
        )
        return self

    def predict(self, user_index):
        return self.model.predict(user_index)
