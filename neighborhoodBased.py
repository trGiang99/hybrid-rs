import pandas as pd
import numpy as np

from math import sqrt

from scipy import sparse
from scipy.sparse.linalg import norm


class kNN:
    """Reimplementation of kNN argorithm.

    Args:
            data: Training data
            k (int): Number of neibors use in prediction
            distance (str, optional): Distance function. Defaults to "cosine".
            uuCF (boolean, optional): Assign to 1 if using user-based CF, 0 if using item-based CF. Defaults to 0.
            normalize (str, optional): Normalization method. Defaults to "none".
    """
    def __init__(self, data, k, distance="cosine", uuCF=0, normalize="none"):
        self.k = k
        self.ultility = data

        self.__supported_disc_func = ["cosine", "pearson"]

        assert distance in self.__supported_disc_func, f"Distance function should be one of {self.__supported_disc_func}"
        if distance == "cosine":
            self.__distance = self.__cosine
        elif distance == "pearson":
            self.__distance = self.__pcc

        self.uuCF = uuCF

        self.__normalize_method = ["none", "mean", "baseline"]
        assert normalize in self.__normalize_method, f"Normalize method should be one of {self.__normalize_method}"
        if normalize == "mean":
            self.__normalize = self.__mean_normalize
        elif normalize == "baseline":
            self.__normalize = self.__baseline
        else:
            self.__normalize = False

        # Set up the similarity matrix
        if(uuCF):
            self.S = sparse.lil_matrix((self.ultility.shape[0], self.ultility.shape[0]), dtype='float64')
        else:
            self.S = sparse.lil_matrix((self.ultility.shape[1], self.ultility.shape[1]), dtype='float64')

    def fit(self):
        """Fit data (ultility matrix) into the model.
        """
        if(self.__normalize):
            print("Normalizing the utility matrix ...")
            self.__normalize()
            print("Done.")

        print('Computing similarity matrix ...')
        self.__distance()
        print('Done.')

        # Convert to Compressed Sparse Row format for faster arithmetic operations.
        self.S.tocsr()

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
            users_rated_i = users_rated_i[users_rated_i != u]

            sim = []
            for user_rated_i in users_rated_i:
                sim.append(self.S[u, user_rated_i]) if self.S[u, user_rated_i] else sim.append(self.S[user_rated_i, u])

            sim = np.array(sim)

            # Sort similarity list in descending
            k_nearest_users = np.argsort(sim)[-self.k:]

            # Get first k users or all if number of similar users smaller than k
            sim = sim[k_nearest_users]
            ratings = np.array([self.ultility[v, i] for v in users_rated_i[k_nearest_users]])

            prediction = np.sum(sim * ratings) / (np.abs(sim).sum() + 1e-8)
        # Item based CF
        else:
            # Find items that have been rated by user u beside item i
            items_ratedby_u = self.ultility.getrow(u).nonzero()[1]
            items_ratedby_u = items_ratedby_u[items_ratedby_u != i]

            sim = []
            for item_ratedby_u in items_ratedby_u:
                sim.append(self.S[i, item_ratedby_u]) if self.S[i, item_ratedby_u] else sim.append(self.S[item_ratedby_u, i])

            sim = np.array(sim)

            # Sort similarity list in descending
            k_nearest_items = np.argsort(sim)[-self.k:]

            # Get first k items or all if number of similar items smaller than k
            sim = sim[k_nearest_items]
            ratings = np.array([self.ultility[u, j] for j in items_ratedby_u[k_nearest_items]])

            prediction = np.sum(sim * ratings) / (np.abs(sim).sum() + 1e-8)

        if (self.__normalize):
            return prediction + self.mu[u]
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
            pred = self.predict(test_data["user_id"][n], test_data["item_id"][n])
            # print(f"Predicting {test_data['user_id'][n]},{test_data['item_id'][n]}: {pred} - {test_data['rating'][n]}")
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
                for v in users[(uidx+1):]:
                    sum_ratings = self.ultility[u,:] * self.ultility[v,:].transpose()
                    if (not sum_ratings):
                        self.S[u,v] = 0
                        continue

                    norm2_ratings_u = norm(self.ultility[u,:], 'fro')
                    norm2_ratings_v = norm(self.ultility[v,:], 'fro')

                    self.S[u,v] = (sum_ratings / (norm2_ratings_u * norm2_ratings_v)).data[0]

        # Item based CF
        else:
            items = np.unique(self.ultility.nonzero()[1])

            for iidx, i in enumerate(items):
                for j in items[(iidx+1):]:
                    sum_ratings = self.ultility[:,i].transpose() * self.ultility[:,j]
                    if (not sum_ratings):
                        self.S[i,j] = 0
                        continue

                    norm2_ratings_i = norm(self.ultility[:,i], 'fro')
                    norm2_ratings_j = norm(self.ultility[:,j], 'fro')

                    self.S[i,j] = (sum_ratings / (norm2_ratings_i * norm2_ratings_j)).data[0]

    def __pcc(self, i, j):
        pass

    def __mean_normalize(self):
        """Normalize the ultility matrix.
        This method only normalize the data base on the mean of ratings.
        Any unrated item will remain the same.
        """
        tot = np.array(self.ultility.sum(axis=1).squeeze())[0]
        cts = np.diff(self.ultility.indptr)
        cts[cts == 0] = 1       # Avoid dividing by 0 resulting nan.

        # Mean ratings of each users.
        self.mu = tot / cts

        # Diagonal matrix with the means on the diagonal.
        d = sparse.diags(self.mu, 0)

        # A matrix that is like Ultility, but has 1 at the non-zero position instead of the ratings.
        b = self.ultility.copy()
        b.data = np.ones_like(b.data)

        # d*b = Mean matrix - a matrix with the means of each row at the non-zero position
        # Subtract the mean matrix to get the normalize data.
        self.ultility -= d*b

    def __baseline(self):
        pass
