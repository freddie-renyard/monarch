import numpy as np
from math import sqrt

def compute_rmse(data_1, data_2):
    """Computes the root mean squared error between
    two datasets. Data is clipped to data 1's length.
    """

    data_1 = np.array(data_1).flatten()
    data_2 = np.array(data_2).flatten()

    data_len = np.shape(data_1)[0]

    squares = (data_2[:data_len] - data_1) ** 2
    rmse = sqrt(np.sum(squares) / data_len)

    return rmse