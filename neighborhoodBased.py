import pandas as pd
import numpy as np

from math import sqrt
from scipy.sparse.linalg import norm


class kNN:
    """Implementation of kNN argorithm.

    Args:
            data: training data
            k (int): number of neibors use in prediction
            distance (str, optional): Distance function. Defaults to "cosine".
            baseline (int, optional): assign to 1 if using basline estimate, 0 if not. Defaults to 0.
    """
    def __init__(self, data, k, distance="cosine", baseline=0):
        self.k = k
        self.data = data

        self.supported_disc_func = ["cosine", "pcc", "cubebCos", "cubedPcc"]

        assert distance in self.supported_disc_func, f"distance function should be one of {self.supported_disc_func}"
        self.distance = distance

        self.baseline = baseline

    def __cosine(self, i, j, user_base=0):
        """Calculate cosine simularity score between object i and object j.
        Object i and j could be item or user, but must be the same.

        Args:
            i (int): index of object i
            j (int): index of object j
            user_base (boolean, optional): assign to 1 if using user-based CF, 0 if using item-based CF. Defaults to 0.

        Returns:
            float: cosine simularity score
        """
        if(user_base):
            sum_ratings = self.data[i,:] * self.data[j,:].transpose()

            norm2_ratings_i = norm(self.data[i,:], 'fro')
            norm2_ratings_j = norm(self.data[j,:], 'fro')
        else:
            sum_ratings = self.data[:,i].transpose() * self.data[:,j]

            norm2_ratings_i = norm(self.data[:, i], 'fro')
            norm2_ratings_j = norm(self.data[:, j], 'fro')

        return sum_ratings / (norm2_ratings_i * norm2_ratings_j)
