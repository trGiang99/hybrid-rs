import numpy as np
import pandas as pd
from scipy import sparse
import time

from .utils import timer

from .kNN import kNN
from .svd import svd
from .helper import _run_epoch, _get_simratings_tensor


class hybrid(svd, kNN):
    def __init__(self, knn_options, svd_options):

        svd.__init__(self, **svd_options)

        kNN.__init__(self, **knn_options)

    def fit(self, train_data, movie_genome=None, user_genome=None, early_stopping=False, shuffle=False, min_delta=0.001):
        kNN.fit(self, train_data=None, genome=movie_genome)

        self.__fit_svd_with_knn(
            train_data=train_data,
            S=self.S,
            k=self.k,
            i_factor=movie_genome,
            u_factor=user_genome,
            early_stopping=early_stopping,
            min_delta=min_delta
        )

    @timer(text='\nTraining took ')
    def __fit_svd_with_knn(self, train_data, S=None, k=None, i_factor=None, u_factor=None, early_stopping=False, min_delta=0.001):
        """Learns model weights using SGD algorithm.
        Args:
            X (pandas DataFrame): training set, must have `u_id` for user id,
                `i_id` for item id and `rating` columns.
            S (numpy array): similarity matrix.
            k (int): k nearest neighbors.
            i_factor (pandas DataFrame, defaults to `None`): initialization for Qi. The dimension should match self.factor
            u_factor (pandas DataFrame, defaults to `None`): initialization for Pu. The dimension should match self.factor
            early_stopping (boolean): whether or not to stop training based on
                a validation monitoring.
            min_delta (float, defaults to .001): minimun delta to arg for an
                improvement.
        Returns:
            self (SVD object): the current fitted object.
        """
        svd.fit(self,
            X=train_data,
            i_factor=i_factor,
            u_factor=u_factor,
            early_stopping=False, shuffle=False
        )

        X = self._preprocess_data(train_data, train=False)
        print("Getting similarity scores and true ratings for each training point.")
        start_cal_sim = time.time()
        self.simratings_tensor = _get_simratings_tensor(X, self.S, self.k)
        finish_cal_sim = time.time()
        print(f"Done. Took {(finish_cal_sim - start_cal_sim):.4f}s")

        pu = self.pu
        qi = self.qi
        bu = self.bu
        bi = self.bi

        print('Start training with KNN...')
        for epoch_ix in range(self.n_epochs):
            start = self._on_epoch_begin(epoch_ix)

            pu, qi, bu, bi, train_loss = _run_epoch(
                                X, pu, qi, bu, bi, self.simratings_tensor, self.global_mean, self.n_factors,
                                self.lr_pu/10, self.lr_qi/10, self.lr_bu/10, self.lr_bi/10,
                                self.reg_pu/10, self.reg_qi/10, self.reg_bu/10, self.reg_bi/10
                            )
            self._on_epoch_end(start, train_loss=train_loss)

        self.pu = pu
        self.qi = qi
        self.bu = bu
        self.bi = bi
        print("Done.")

        return self


    def predict(self, train_data, test_data, mode="train", clip=True):
        """Returns estimated ratings of several given user/item pairs.
        Args:
            train_data (pandas DataFrame): training set.
            test_data (pandas DataFrame): storing all user/item pairs we want to
                predict the ratings. Must contains columns labeled `u_id` and
                `i_id`.
            mode (string, "train" or "full"): "train" mode use only training data in kNN, while "full" mode use full dataset.
                                              Be careful when using "full" mode, the training set need to contain all movie that exist in the dataset, at least 1 each.
            clip (boolean, default is `True`): whether to clip the prediction
                or not.
        Returns:
            predictions: list, storing all predictions of the given user/item
                pairs.
        """
        predictions = []

        if mode == "train":
            X = self._preprocess_data(train_data, train="False")
        else:
            X = self._preprocess_data(pd.concat([train_data, test_data]), train="False")

        for u_id, i_id in zip(test_data['u_id'], test_data['i_id']):
            user_known, item_known = False, False
            pred = self.global_mean

            if u_id in self.user_dict:
                user_known = True
                u_ix = self.user_dict[u_id]
                pred += self.bu[u_ix]

            if i_id in self.item_dict:
                item_known = True
                i_ix = self.item_dict[i_id]
                pred += self.bi[i_ix]

            if user_known and item_known:
                pred += np.dot(self.pu[u_ix], self.qi[i_ix])

                # Find items that have been rated by user u beside item i
                items_ratedby_u = np.array([
                    int(item) for user, item in zip(X[:,0], X[:,1])
                        if item != i_ix and user == u_ix
                ])

                sim = np.array([
                    self.S[i_ix, item_ratedby_u] if self.S[i_ix, item_ratedby_u] else self.S[item_ratedby_u, i_ix]
                        for item_ratedby_u in items_ratedby_u
                ])

                # Sort similarity list in descending
                k_nearest_items = np.argsort(sim)[-self.k:]

                # Get first k items or all if number of similar items smaller than k
                sim = sim[k_nearest_items]
                sim_ratings = np.array([
                    X[np.where((X[:,1] == j) & (X[:,0] == u_ix))[0][0], 2]
                        for j in items_ratedby_u[k_nearest_items]
                ])
                sim_ratings -= pred

                pred += np.sum(sim * sim_ratings) / (np.abs(sim).sum() + 1e-8)

            if clip:
                pred = self.max_rating if pred > self.max_rating else pred
                pred = self.min_rating if pred < self.min_rating else pred

            predictions.append(pred)

        return predictions
