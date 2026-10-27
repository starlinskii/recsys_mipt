import numpy as np


class CustomTargetALS:
    def fit(
        self,
        observed_matrix,
        target_matrix,
        factors=16,
        iterations=8,
        alpha=20,
        regularization=0.1,
    ):
        observed_matrix = observed_matrix.tocsr()
        target_matrix = target_matrix.tocsr()
        num_users, num_items = observed_matrix.shape
        rng = np.random.default_rng(0)
        self.user_factors = rng.normal(0, 0.01, size=(num_users, factors))
        self.item_factors = rng.normal(0, 0.01, size=(num_items, factors))
        identity = np.eye(factors)

        for _ in range(iterations):
            self.user_factors = self.update_factors(
                observed_matrix,
                target_matrix,
                self.item_factors,
                identity,
                alpha,
                regularization,
            )
            self.item_factors = self.update_factors(
                observed_matrix.T.tocsr(),
                target_matrix.T.tocsr(),
                self.user_factors,
                identity,
                alpha,
                regularization,
            )
        return self

    def update_factors(
        self,
        observed_matrix,
        target_matrix,
        fixed_factors,
        identity,
        alpha,
        regularization,
    ):
        num_rows = observed_matrix.shape[0]
        factors = fixed_factors.shape[1]
        result = np.zeros((num_rows, factors))
        base_matrix = fixed_factors.T @ fixed_factors

        for row_number in range(num_rows):
            start = observed_matrix.indptr[row_number]
            end = observed_matrix.indptr[row_number + 1]
            item_indices = observed_matrix.indices[start:end]
            targets = target_matrix.data[start:end]
            item_factors = fixed_factors[item_indices]

            matrix = base_matrix + alpha * (item_factors.T @ item_factors)
            matrix += regularization * identity
            vector = item_factors.T @ ((1 + alpha) * targets)
            result[row_number] = np.linalg.solve(matrix, vector)

        return result

    def predict(self, user_index):
        return self.user_factors[user_index] @ self.item_factors.T
