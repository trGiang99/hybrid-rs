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

        if(uuCF):
            self.S = np.zeros([self.ultility.shape[0], self.ultility.shape[0]])
        else:
            self.S = np.zeros([self.ultility.shape[1], self.ultility.shape[1]])

    def fit(self):
        """Calculate the similarity matrix
        """
        self.distance()

        # Turn the triangle matrix self.S into a full matrix
        full_S = self.S.T + self.S
        np.fill_diagonal(full_S, np.diag(self.S))
        self.S = full_S

    def predict(self, u, i):
        """Predict the rating of user u for item i

        Args:
            u (int): index of user
            i (int): index of item
        """
        # If there's already a rating
        if(self.ultility[u,i]):
            print (f"User {u} has rated movie {i} already.")
            return
        # User based CF
        if (self.uuCF):
            # Find users that have rated item i beside user u
            user_rated_i = self.ultility.getcol(i).nonzero()[0]

            sim = self.S[u, user_rated_i]

            # Sort similarity list in descending
            k_nearest_users = np.argsort(sim)[-self.k:]

            # Get first k users or all if number of similar users smaller than k
            sim = sim[k_nearest_users]
            ratings = np.array([self.ultility[v, i] for v in user_rated_i[k_nearest_users]])

            prediction = np.sum(sim * ratings) / (np.sum(sim) + 1e-8)
        # Item based CF
        else:
            # Find items that have been rated by user u beside item i
            items_ratedby_u = self.ultility.getrow(u).nonzero()[1]

            sim = self.S[i, items_ratedby_u]

            # Sort similarity list in descending
            k_nearest_items = np.argsort(sim)[-self.k:]

            # Get first k items or all if number of similar items smaller than k
            sim = sim[k_nearest_items]
            ratings = np.array([self.ultility[u, j] for j in user_rated_i[k_nearest_items]])

            prediction = np.sum(sim * ratings) / (np.sum(sim) + 1e-8)

        return prediction

    def __recommend(self, u):
        """Determine all items should be recommended for user u. (uuCF =1)
        or all users who might have interest on item u (uuCF = 0)
        The decision is made based on all i such that: self.pred(u, i) > 0.
        Suppose we are considering items which have not been rated by u yet.

        Args:
            u (int): user that we are recommending

        Returns:
            list: a list of movie that might suit user u
        """
        pass

    def rmse(self, test_data):
        """Calculate Root Mean Squared Error on the test data

        Args:
            test_data (DataFrame): testing data

        Returns:
            float: RMSE
        """
        squared_error = 0
        n_test_ratings = test_data.shape[0]

        for n in range(n_test_ratings):
            # print(f"Predicting {test_data['user_id'][n]},{test_data['item_id'][n]}")
            pred = self.predict(test_data["user_id"][n], test_data["item_id"][n])
            squared_error += (pred - test_data["rating"][n])**2

        return np.sqrt(squared_error/n_test_ratings)

    def __cosine(self):
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
            n_users = self.ultility.shape[0] - 1
            for i in range(1, n_users + 1):
                for j in range(i, n_users + 1):
                    sum_ratings = self.ultility[i,:] * self.ultility[j,:].transpose()
                    if (not sum_ratings):
                        self.S[i][j] = 0
                        continue

                    norm2_ratings_i = norm(self.ultility[i,:], 'fro')
                    norm2_ratings_j = norm(self.ultility[j,:], 'fro')

                    self.S[i][j] = (sum_ratings / (norm2_ratings_i * norm2_ratings_j)).data[0]
        # Item based CF
        else:
            n_items = self.ultility.shape[1] - 1
            for i in range(1, n_items + 1):
                for j in range(i, n_items + 1):
                    sum_ratings = self.ultility[:,i].transpose() * self.ultility[:,j]
                    if (not sum_ratings):
                        return 0

                    norm2_ratings_i = norm(self.ultility[:,i], 'fro')
                    norm2_ratings_j = norm(self.ultility[:,j], 'fro')

                    self.S[i][j] = (sum_ratings / (norm2_ratings_i * norm2_ratings_j)).data[0]

    def __pcc(self, i, j):
        pass
