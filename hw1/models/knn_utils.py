import numpy as np
from scipy import sparse


def keep_top_neighbors(similarity, n_neighbors):
    rows = []
    columns = []
    values = []

    for row_number in range(similarity.shape[0]):
        start = similarity.indptr[row_number]
        end = similarity.indptr[row_number + 1]
        row_values = similarity.data[start:end]
        row_columns = similarity.indices[start:end]

        if len(row_values) > n_neighbors:
            selected = np.argpartition(row_values, -n_neighbors)[-n_neighbors:]
            row_values = row_values[selected]
            row_columns = row_columns[selected]

        rows.extend([row_number] * len(row_values))
        columns.extend(row_columns)
        values.extend(row_values)

    return sparse.csr_matrix(
        (values, (rows, columns)),
        shape=similarity.shape,
    )

