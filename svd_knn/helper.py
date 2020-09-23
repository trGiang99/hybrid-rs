import numpy as np
from numba import njit


@njit
def _shuffle(X):
    np.random.shuffle(X)
    return X


@njit
def _run_epoch(X, pu, qi, bu, bi, sim_ratings, global_mean, n_factors, lr_pu, lr_qi, lr_bu, lr_bi, reg_pu, reg_qi, reg_bu, reg_bi):
    """Runs an epoch, updating model weights (pu, qi, bu, bi).
    Using Gradient Descent instead of Stochastic Gradient Descent.
    Args:
        X (numpy array): training set.
        pu (numpy array): users latent factor matrix.
        qi (numpy array): items latent factor matrix.
        bu (numpy array): users biases vector.
        bi (numpy array): items biases vector.
        S (numpy array): similarity matrix.
        k: (numpy array): k nearest neighbors.
        global_mean (float): ratings arithmetic mean.
        n_factors (int): number of latent factors.
        lr (float): learning rate.
        reg (float): regularization factor.
    Returns:
        pu (numpy array): users latent factor matrix updated.
        qi (numpy array): items latent factor matrix updated.
        bu (numpy array): users biases vector updated.
        bi (numpy array): items biases vector updated.
        train_loss (float): training loss.
    """
    pred = np.zeros(X.shape[0])

    # Predict all existing ratings in the training set using SVD fomular
    for i in range(X.shape[0]):
        user, item = int(X[i, 0]), int(X[i, 1])

        # Predict current rating
        pred[i] = global_mean + bu[user] + bi[item]

        for factor in range(n_factors):
            pred[i] += pu[user, factor] * qi[item, factor]

    # Predict ratings by intergrating KNN into SVD
    temp_pred = np.zeros_like(pred)
    for idx in range(pred.shape[0]):
        indices = sim_ratings[idx, 0, :]
        sim_scores = sim_ratings[idx, 1, :]
        ratings = sim_ratings[idx, 2, :]
        temp_k_pred = np.array([pred[int(i)] for i in indices])
        temp_pred[idx] = pred[idx] + (np.sum(sim_scores * (ratings - temp_k_pred)) / (np.abs(sim_scores).sum() + 1e-8))
    pred = temp_pred

    err = X[:,2] - pred

    for i in range(X.shape[0]):
        user, item = int(X[i, 0]), int(X[i, 1])
        # Update biases
        bu[user] += lr_bu * (err[i] - reg_bu * bu[user])
        bi[item] += lr_bi * (err[i] - reg_bi * bi[item])

        # Update latent factors
        for factor in range(n_factors):
            puf = pu[user, factor]
            qif = qi[item, factor]

            pu[user, factor] += lr_pu * (err[i] * qif - reg_pu * puf)
            qi[item, factor] += lr_qi * (err[i] * puf - reg_qi * qif)

    train_loss = np.square(err).mean()
    return pu, qi, bu, bi, train_loss


@njit
def _get_simratings_tensor(X, S, k):
    """Get all similarity scores and ratings of similar item for
    every (user, item) pair in the training set.

    Args:
        X (ndarray): training set
        S (ndarray): similarity matrix
        k (int): k nearest neighbors

    Returns:
        sim_ratings (ndarray): Tensor contains all needed information.
                               First column: index of the rating in third corresponding to the training set
                               Second column: similarity scores of k most similar items to item i_id rated by user u_id
                               Third column: the ratings of of k most similar items to item i_id rated by user u_id
    """
    sim_ratings = np.zeros((X.shape[0], 3, k))

    for train_index in range(X.shape[0]):
        user, item = int(X[train_index, 0]), int(X[train_index, 1])

        list_index = []
        list_sims = []
        list_ratings = []
        for u_id, i_id, rating_u in zip(X[:,0], X[:,1], X[:,2]):
            if i_id != item and u_id == user:
                item_ratedby_u = int(i_id)
                list_index.append(item_ratedby_u)
                # Get similarity score to items that are also rated by user u
                list_sims.append(S[item, item_ratedby_u] if S[item, item_ratedby_u] else S[item_ratedby_u, item])
                list_ratings.append(rating_u)
        indices = np.array(list_index)
        sims = np.array(list_sims)
        ratings = np.array(list_ratings)

        # Sort similarity list in descending and get k first indices
        k_nearest_items = np.argsort(sims)[-k:]

        # Get first k items or all if number of similar items smaller than k
        for idx, k_nearest_item in enumerate(k_nearest_items):
            sim_ratings[train_index, 0, idx] = indices[k_nearest_item]
            sim_ratings[train_index, 1, idx] = sims[k_nearest_item]
            sim_ratings[train_index, 2, idx] = ratings[k_nearest_item]

    return sim_ratings
