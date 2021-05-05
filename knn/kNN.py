import pandas as pd
import numpy as np
from math import sqrt

import progressbar

from scipy import sparse
import heapq
from scipy.sparse.linalg import norm

from utils import timer
from .sim import _cosine, _pcc, _cosine_genome, _pcc_genome
from .knn_helper import _baseline_sgd, _predict


class kNN:
    """Reimplementation of kNN argorithm.

    Args:
        k (int): Number of neibors use in prediction
        min_k (int): The minimum number of neighbors to take into account for aggregation. If there are not enough neighbors, the neighbor aggregation is set to zero
        distance (str, optional): Distance function. Defaults to "cosine".
        uuCF (boolean, optional): Assign to 1 if using user-based CF, 0 if using item-based CF. Defaults to 0.
        normalize (str, optional): Normalization method. Defaults to "none".
    """
    def __init__(self, k, min_k=1, distance="cosine", uuCF=0, normalize="none"):
        self.k = k
        self.min_k = min_k

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
        """Fit data (utility matrix) into the model.

        Args:
            data (scipy.sparse.csr_matrix): Training data.
            genome (ndarray): Movie genome scores from MovieLens 20M.
        """
        self.X = train_data


        self.user_list = self.X[:, 0]
        self.item_list = self.X[:, 1]

        if(self.__normalize):
            print("Normalizing the utility matrix ...")
            self.__normalize()

        print('Computing similarity matrix ...')
        if self.__distance == "cosine":
            if genome is not None:
                self.S = _cosine_genome(genome)
            else:
                self.S = _cosine(self.X, self.uuCF)
        elif self.__distance == "pearson":
            if genome is not None:
                self.S = _pcc_genome(genome)
            else:
                self.S = _pcc(self.X, self.uuCF)

        self.utility = sparse.csr_matrix((
            self.X[:, 2],
            (self.X[:, 0].astype(int), self.X[:, 1].astype(int))
        ))

    def predict(self, u_id, i_id):
        """Predict the rating of user u for item i

        Args:
            u_id (int): index of user
            i_id (int): index of item
        """
        if(self.utility[u_id,i_id]):
            print (f"User {u} has already rated movie {i}.")
            return

        if self.uuCF:
            col_u = self.utility.getcol(i_id)

            users_rated_i = col_u.nonzero()[0]
            ratings = col_u.data
            sim = [self.S(u_id, u) for u in users_rated_i]

            neighbors = list(zip(users_rated_i, sim, ratings))
        else:
            row_i = self.utility.getrow(u_id)

            items_ratedby_u = row_i.nonzero()[0]
            ratings = row_i.data
            sim = [self.S[i_id, i] for i in items_ratedby_u]

            neighbors = list(zip(items_ratedby_u, sim, ratings))

        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[1])

        if self.__normalize == self.__baseline:
            pred = self.global_mean

            user_known, item_known = False, False
            if u_id in self.user_list:
                user_known = True
                pred += self.bu[u_id]
            if i_id in self.item_list:
                item_known = True
                pred += self.bi[i_id]

            if not (user_known and item_known):
                return pred
            pred += _predict(u_id, i_id, k_neighbors, self.min_k, self.uuCF, self.global_mean, self.bu, self.bi)

        elif self.__normalize == self.__mean_normalize:
            return pred + self.mu[u]

        return pred

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

    @timer("Time for predicting: ")
    def rmse(self, test_data):
        """Calculate Root Mean Squared Error on the test data

        Args:
            test_data (DataFrame): testing data

        Returns:
            float: RMSE
        """
        squared_error = 0
        n_test_ratings = test_data.shape[0]

        bar = progressbar.ProgressBar(maxval=3957876, widgets=[progressbar.Bar(), ' ', progressbar.Percentage()])
        bar.start()
        for n in range(n_test_ratings):
            pred = self.predict(test_data[n, 0].astype(int), test_data[n, 1].astype(int))
            squared_error += (pred - test_data[n, 2])**2
            bar.update(n+1)
        bar.finish()

        return np.sqrt(squared_error/n_test_ratings)

    def __mean_normalize(self):
        """Normalize the utility matrix.
        This method only normalize the data base on the mean of ratings.
        Any unrated item will remain the same.
        """
        tot = np.array(self.X.sum(axis=1).squeeze())[0]
        cts = np.diff(self.X.indptr)
        cts[cts == 0] = 1       # Avoid dividing by 0 resulting nan.

        # Mean ratings of each users.
        self.mu = tot / cts

        # Diagonal matrix with the means on the diagonal.
        d = sparse.diags(self.mu, 0)

        # A matrix that is like Utility, but has 1 at the non-zero position instead of the ratings.
        b = self.X.copy()
        b.data = np.ones_like(b.data)

        # d*b = Mean matrix - a matrix with the means of each row at the non-zero position
        # Subtract the mean matrix to get the normalize data.
        self.X -= d*b

    def __baseline(self):
        """Normalize the utility matrix.
        Compute the baseline estimate for all user and movie using the following fomular.
        b_{ui} = \mu + b_u + b_i
        """
        self.global_mean = np.mean(self.X[:, 2])
        n_users = len(np.unique(self.X[:, 0]))
        n_items = len(np.unique(self.X[:, 1]))

        self.bu, self.bi = _baseline_sgd(self.X, self.global_mean, n_users, n_items)
