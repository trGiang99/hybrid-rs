import pandas as pd
import numpy as np

import progressbar

from scipy import sparse
from scipy.sparse.linalg import norm

from utils import timer
from .sim import _cosine, _pcc, _cosine_genome, _pcc_genome
from .knn_helper import _baseline_sgd, _predict, _predict_mean, _predict_baseline


class kNN:
    """Reimplementation of kNN argorithm.

    Args:
        k (int): Number of neibors use in prediction
        min_k (int): The minimum number of neighbors to take into account for aggregation. If there are not enough neighbors, the neighbor aggregation is set to zero
        distance (str, optional): Distance function. Defaults to `"cosine"`.
        uuCF (boolean, optional): True if using user-based CF, False if using item-based CF. Defaults to `False`.
        normalize (str, optional): Normalization method. Defaults to `None`.
        verbose (boolean): Show predicting progress. Defaults to `False`.
        awareness_constrain (boolean): Only for Baseline model (besides, considered `False`). If `True`, the model must aware of all users and items in the test set, which means that these users and items are in the train set as well. This constrain helps speed up the predicting process (by 1.5 times) but if a user of an item is unknown, kNN will fail to give prediction. Defaults to `True`.
    """
    def __init__(self, k, min_k=1, distance="cosine", uuCF=False, normalize="none", verbose=False, awareness_constrain=True):
        self.k = k
        self.min_k = min_k

        self.__supported_disc_func = ["cosine", "pearson"]
        assert distance in self.__supported_disc_func, f"Distance function should be one of {self.__supported_disc_func}"
        self.__distance = distance

        self.uuCF = uuCF

        self.verbose = verbose
        self.awareness_constrain = awareness_constrain

        self.__normalize_method = ["none", "mean", "baseline"]
        assert normalize in self.__normalize_method, f"Normalize method should be one of {self.__normalize_method}"
        if normalize == "mean":
            self.__normalize = self.__mean_normalize
        elif normalize == "baseline":
            self.__normalize = self.__baseline
        else:
            self.__normalize = False

    def fit(self, train_data, genome=None, sim=None):
        """Fit data (utility matrix) into the model.

        Args:
            data (scipy.sparse.csr_matrix): Training data.
            genome (ndarray): Movie genome scores from MovieLens 20M. Defaults to "none".
            sim (ndarray): Pre-calculate similarity matrix.  Defaults to "none".
        """
        self.X = train_data

        if self.uuCF:
            X[:, [0, 1]] = X[:, [1, 0]]     # Swap user_id column to movie_id column

        self.x_list = np.unique(self.X[:, 0])       # For iiCF, x -> user
        self.y_list = np.unique(self.X[:, 1])       # For iiCF, y -> item

        self.n_x = len(self.x_list)
        self.n_y = len(self.y_list)

        if(self.__normalize):
            print("Normalizing the utility matrix ...")
            self.__normalize()

        self.utility = sparse.csr_matrix((
            self.X[:, 2],
            (self.X[:, 0].astype(int), self.X[:, 1].astype(int))
        ))

        if sim is None:
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
        else:
            self.S = sim

        # List where element `i` is ndarray of `(xs, rating)` where `xs` is all x that rated y and the ratings.
        self.y_rated = []
        print("Listing all items rated by each users (or vice versa if uuCF) ...")
        for id in range(self.n_x):
            row_i = self.utility.getrow(id)
            ys_rated_x = row_i.nonzero()[1]
            ratings = row_i.data

            self.y_rated.append(np.dstack((ys_rated_x, ratings))[0])

    @timer("Time for predicting: ")
    def predict(self, X):
        """Returns estimated ratings of several given user/item pairs.
        Args:
            X (pandas DataFrame): storing all user/item pairs we want to predict
            the ratings. Must contains columns labeled `u_id` and `i_id`.
        Returns:
            predictions (ndarray): Storing all predictions of the given user/item pairs.
        """
        self.predictions = []
        self.ground_truth = X[:, 2]
        n_pairs = X.shape[0]

        print(f"Predicting {n_pairs} pairs of user-item ...")

        if self.verbose:
            bar = progressbar.ProgressBar(maxval=n_pairs, widgets=[progressbar.Bar(), ' ', progressbar.Percentage()])
            bar.start()
            for pair in range(n_pairs):
                self.predictions.append(self.predict_pair(X[pair, 0].astype(int), X[pair, 1].astype(int)))
                bar.update(pair + 1)
            bar.finish()
        else:
            for pair in range(n_pairs):
                self.predictions.append(self.predict_pair(X[pair, 0].astype(int), X[pair, 1].astype(int)))

        self.predictions = np.array(self.predictions)
        return self.predictions

    def predict_pair(self, x_id, y_id):
        """Predict the rating of user u for item i

        Args:
            x_id (int): index of x (For iiCF, x -> user)
            y_id (int): index of y (For iiCF, y -> item)

        Returns:
            pred (float): prediction of the given user/item pair.
        """
        if self.__normalize == self.__baseline:
            if self.awareness_constrain:
                pred = self.global_mean

                x_known, y_known = False, False
                if x_id in self.x_list:
                    x_known = True
                    pred += self.bx[x_id]
                if y_id in self.y_list:
                    y_known = True
                    pred += self.by[y_id]

                if not (x_known and y_known):
                    return pred

            pred = _predict_baseline(x_id, y_id, self.y_rated[x_id], self.S, self.k, self.min_k, self.global_mean, self.bx, self.by)
            return pred

        else:
            x_known, y_known = False, False

            if x_id in self.x_list:
                x_known = True
            if y_id in self.y_list:
                y_known = True

            if not (x_known and y_known):
                if uuCF:
                    print(f"Can not predict rating of user {y_id} for item {x_id}.")
                else:
                    print(f"Can not predict rating of user {x_id} for item {y_id}.")
                return

            if self.__normalize == self.__mean_normalize:
                pred += _predict_mean(x_id, y_id, self.y_rated[x_id], self.mu, self.S, self.k, self.min_k)
                return pred + self.mu[u]
            else:
                return _predict(x_id, y_id, self.y_rated[x_id], self.S, self.k, self.min_k)

    def __recommend(self, u):
        """Determine all items should be recommended for user u. (uuCF = 1)
        or all users who might have interest on item u (uuCF = 0)
        The decision is made based on all i such that: self.pred(u, i) > 0.
        Suppose we are considering items which have not been rated by u yet.
        NOT YET IMPLEMENTED...

        Args:
            u (int): user that we are recommending

        Returns:
            list: a list of movie that might suit user u
        """
        pass

    def rmse(self):
        """Calculate Root Mean Squared Error between the predictions and the ground truth.
        Print the RMSE.
        """
        mse = np.mean((self.predictions - self.ground_truth)**2)
        rmse_ = np.sqrt(mse)
        print(f"RMSE: {rmse_:.5f}")

    def mae(self):
        """Calculate Mean Absolute Error between the predictions and the ground truth.
        Print the MAE.
        """
        mae_ = np.mean(np.abs(self.predictions - self.ground_truth))
        print(f"MAE: {mae_:.5f}")

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
        self.global_mean = np.mean(self.X[:, 2])

        self.bx, self.by = _baseline_sgd(self.X, self.global_mean, self.n_x, self.n_y)
