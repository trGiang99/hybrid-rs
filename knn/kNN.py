import pandas as pd
import numpy as np
from math import sqrt

from scipy import sparse
from scipy.sparse.linalg import norm

from utils import timer
from .sim import _cosine, _pcc, _cosine_genome, _pcc_genome
from .knn_helper import _baseline_sgd


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
        self.utility = train_data

        self.user_list = self.utility[:, 0]
        self.item_list = self.utility[:, 1]

        if(self.__normalize):
            print("Normalizing the utility matrix ...")
            self.__normalize()

        print('Computing similarity matrix ...')
        if self.__distance == "cosine":
            if genome is not None:
                self.S = _cosine_genome(genome)
            else:
                self.S = _cosine(self.utility, self.uuCF)
        elif self.__distance == "pearson":
            if genome is not None:
                self.S = _pcc_genome(genome)
            else:
                self.S = _pcc(self.utility, self.uuCF)

    def predict(self, u_id, i_id):
        """Predict the rating of user u for item i

        Args:
            u (int): index of user
            i (int): index of item
        """
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

        pred += _predict(u_id, i_id, self.utility, self.S, self.k, self.min_k, self.uuCF, self.global_mean, self.bu, self.bi)
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

        for n in range(n_test_ratings):
            pred = self.predict(test_data[n, 0].astype(int), test_data[n, 1].astype(int))
            squared_error += (pred - test_data[n, 2])**2

        return np.sqrt(squared_error/n_test_ratings)

    def __mean_normalize(self):
        """Normalize the utility matrix.
        This method only normalize the data base on the mean of ratings.
        Any unrated item will remain the same.
        """
        tot = np.array(self.utility.sum(axis=1).squeeze())[0]
        cts = np.diff(self.utility.indptr)
        cts[cts == 0] = 1       # Avoid dividing by 0 resulting nan.

        # Mean ratings of each users.
        self.mu = tot / cts

        # Diagonal matrix with the means on the diagonal.
        d = sparse.diags(self.mu, 0)

        # A matrix that is like Utility, but has 1 at the non-zero position instead of the ratings.
        b = self.utility.copy()
        b.data = np.ones_like(b.data)

        # d*b = Mean matrix - a matrix with the means of each row at the non-zero position
        # Subtract the mean matrix to get the normalize data.
        self.utility -= d*b

    def __baseline(self):
        """Normalize the utility matrix.
        Compute the baseline estimate for all user and movie using the following fomular.
        b_{ui} = \mu + b_u + b_i
        """
        self.global_mean = np.mean(self.utility[:, 2])
        n_users = len(np.unique(self.utility[:, 0]))
        n_items = len(np.unique(self.utility[:, 1]))

        self.bu, self.bi = _baseline_sgd(self.utility, self.global_mean, n_users, n_items)
