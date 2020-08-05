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
            uuCF (boolean, optional): assign to 1 if using user-based CF, 0 if using item-based CF. Defaults to 0.
    """
    def __init__(self, data, k, distance="cosine", baseline=0, uuCF=0):
        self.k = k
        self.ultility = data

        self.supported_disc_func = ["cosine", "pcc", "cubebCos", "cubedPcc"]

        assert distance in self.supported_disc_func, f"distance function should be one of {self.supported_disc_func}"
        if distance == "cosine":
            self.distance = self.__cosine

        self.baseline = baseline
        self.uuCF = uuCF

    def __cosine(self, i, j):
        """Calculate cosine simularity score between object i and object j.
        Object i and j could be item or user, but must be the same.

        Args:
            i (int): index of object i
            j (int): index of object j

        Returns:
            float: cosine simularity score
        """
        # User based CF
        if(self.uuCF):
            sum_ratings = self.ultility[i,:] * self.ultility[j,:].transpose()
            if (not sum_ratings):
                return 0

            norm2_ratings_i = norm(self.ultility[i,:], 'fro')
            norm2_ratings_j = norm(self.ultility[j,:], 'fro')

            return (sum_ratings / (norm2_ratings_i * norm2_ratings_j)).data[0]
        # Item based CF
        else:
            sum_ratings = self.ultility[:,i].transpose() * self.ultility[:,j]
            if (not sum_ratings):
                return 0

            norm2_ratings_i = norm(self.ultility[:,i], 'fro')
            norm2_ratings_j = norm(self.ultility[:,j], 'fro')

            return (sum_ratings / (norm2_ratings_i * norm2_ratings_j)).data[0]

    def prediction(self, u, i):
        """Predict the rating of user u for item i

        Args:
            u (int): index of user
            i (int ): index of item
        """
        # If there's already a rating
        if(self.ultility[u,i]):
            print (f"User {u} has rated movie {i} already.")
            return
        # User based CF
        if (self.uuCF):
            # Find users that have rated item i beside user u
            user_rated_i = self.ultility.getcol(i).nonzero()[0]

            sim = dict(zip(user_rated_i, [self.distance(u, v) for v in user_rated_i]))

            # Sort similarity list in descending
            sim = {k: v for k, v in sorted(sim.items(), reverse=True, key=lambda item: item[1])}

            # Get first k users or all if number of users smaller than k
            user_rated_i = [*sim.keys()][:self.k]
            sim = np.array([*sim.values()])[:self.k]

            prediction = np.sum(np.multiply(np.array([self.ultility[v,i] for v in user_rated_i]), sim)) / np.sum(sim)
        # Item based CF
        else:
            # Find items that have been rated by user u beside item i
            items_ratedby_u = self.ultility.getrow(u).nonzero()[1]

            sim = dict(zip(items_ratedby_u, [self.distance(i, j) for j in items_ratedby_u]))

            # Sort similarity list in descending order
            sim = {k: v for k, v in sorted(sim.items(), reverse=True, key=lambda item: item[1])}

            # Get first k users or all if number of users smaller than k
            items_ratedby_u = [*sim.keys()][:self.k]
            sim = np.array([*sim.values()])[:self.k]

            prediction = np.sum(np.multiply(np.array([self.ultility[i,j] for j in items_ratedby_u]), sim)) / np.sum(sim)

        return prediction
