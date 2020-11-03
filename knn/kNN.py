import pandas as pd
import numpy as np
from math import sqrt

from scipy import sparse
from scipy.sparse.linalg import norm

from .utils import timer
from .sim import _cosine, _pcc, _cosine_genome, _pcc_genome


class kNN:
    """Reimplementation of kNN argorithm.

    Args:
            k (int): Number of neibors use in prediction
            distance (str, optional): Distance function. Defaults to "cosine".
            uuCF (boolean, optional): Assign to 1 if using user-based CF, 0 if using item-based CF. Defaults to 0.
            normalize (str, optional): Normalization method. Defaults to "none".
    """
    def __init__(self, k, distance="cosine", uuCF=0, normalize="none"):
        self.k = k

        self.__supported_disc_func = ["cosine", "pearson"]
        assert distance in self.__supported_disc_func, f"Distance function should be one of {self.__supported_disc_func}"
        self.__distance = distance

        self.uuCF = uuCF

        self.__normalize_method = ["none", "mean", "baseline"]
        assert normalize in self.__normalize_method, f"Normalize method should be one of {self.__normalize_method}"
        if normalize == "mean":
            self.__normalize = self.__mean_normalize
        elif normalize == "baseline":
            self.__normalize = self.__baseline
        else:
            self.__normalize = False

    @timer("Runtime: ")
    def fit(self, train_data, genome=None):
        """Fit data (ultility matrix) into the model.

        Args:
            data (scipy.sparse.csr_matrix): Training data.
            genome (ndarray): Movie genome scores from MovieLens 20M.
        """
        self.ultility = train_data
        if(self.__normalize):
            print("Normalizing the utility matrix ...")
            self.__normalize()

        print('Computing similarity matrix ...')
        if self.__distance == "cosine":
            if genome is not None:
                self.S = _cosine_genome(genome)
            else:
                self.S = _cosine(self.ultility, self.uuCF)
        elif self.__distance == "pearson":
            if genome is not None:
                self.S = _pcc_genome(genome)
            else:
                self.S = _pcc(self.ultility, self.uuCF)

    def predict(self, u, i):
        """Predict the rating of user u for item i

        Args:
            u (int): index of user
            i (int): index of item
        """
        # If there's already a rating
        if(self.ultility[u,i]):
            print (f"User {u} has already rated movie {i}.")
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
            pred = self.predict(test_data["u_id"][n].astype(int), test_data["i_id"][n].astype(int))
            squared_error += (pred - test_data["rating"][n])**2

        return np.sqrt(squared_error/n_test_ratings)

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
