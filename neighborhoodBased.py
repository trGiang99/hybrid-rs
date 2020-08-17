import pandas as pd
import numpy as np

from math import sqrt

from scipy import sparse
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

        self.supported_disc_func = ["cosine"]

        assert distance in self.supported_disc_func, f"Distance function should be one of {self.supported_disc_func}"
        if distance == "cosine":
            self.distance = self.__cosine

        self.baseline = baseline
        self.uuCF = uuCF

        if(uuCF):
            self.S = sparse.lil_matrix((self.ultility.shape[0], self.ultility.shape[0]))
        else:
            self.S = sparse.lil_matrix((self.ultility.shape[1], self.ultility.shape[1]))

    def fit(self):
        """Calculate the similarity matrix
        """
        self.distance()

        self.S.tocsr()      # Convert to Compressed Sparse Row format for faster arithmetic operations.

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
            users_rated_i = self.ultility.getcol(i).nonzero()[0]

            sim = []
            for user_rated_i in users_rated_i:
                sim.append(self.S[u, user_rated_i]) if self.S[u, user_rated_i] else sim.append(self.S[user_rated_i, u])

            sim = np.array(sim)

            # Sort similarity list in descending
            k_nearest_users = np.argsort(sim)[-self.k:]

            # Get first k users or all if number of similar users smaller than k
            sim = sim[k_nearest_users]
            ratings = np.array([self.ultility[v, i] for v in users_rated_i[k_nearest_users]])

            prediction = np.sum(sim * ratings) / (np.sum(sim) + 1e-8)
        # Item based CF
        else:
            # Find items that have been rated by user u beside item i
            items_ratedby_u = self.ultility.getrow(u).nonzero()[1]

            sim = []
            for item_ratedby_u in items_ratedby_u:
                sim.append(self.S[i, item_ratedby_u]) if self.S[i, item_ratedby_u] else sim.append(self.S[item_ratedby_u, i])

            sim = np.array(sim)

            # Sort similarity list in descending
            k_nearest_items = np.argsort(sim)[-self.k:]

            # Get first k items or all if number of similar items smaller than k
            sim = sim[k_nearest_items]
            ratings = np.array([self.ultility[u, j] for j in items_ratedby_u[k_nearest_items]])

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
        Assign the similarity score to self.S (similarity matrix).

        Args:
            i (int): index of object i
            j (int): index of object j
        """
        # User based CF
        if(self.uuCF):
            users = np.unique(self.ultility.nonzero()[0])

            for uidx, u in enumerate(users):
                for v in users[uidx:]:
                    sum_ratings = self.ultility[u,:] * self.ultility[v,:].transpose()
                    if (not sum_ratings):
                        self.S[u,v] = 0
                        continue

                    norm2_ratings_u = norm(self.ultility[u,:], 'fro')
                    norm2_ratings_v = norm(self.ultility[v,:], 'fro')

                    self.S[u,v] = (sum_ratings / (norm2_ratings_u * norm2_ratings_v)).data[0]
                    count += 1
            print(count)
        # Item based CF
        else:
            items = np.unique(self.ultility.nonzero()[1])

            for iidx, i in enumerate(items):
                for j in items[iidx:]:
                    sum_ratings = self.ultility[:,i].transpose() * self.ultility[:,j]
                    if (not sum_ratings):
                        self.S[i,j] = 0
                        continue

                    norm2_ratings_i = norm(self.ultility[:,i], 'fro')
                    norm2_ratings_j = norm(self.ultility[:,j], 'fro')

                    self.S[i,j] = (sum_ratings / (norm2_ratings_i * norm2_ratings_j)).data[0]

    def __pcc(self, i, j):
        pass
