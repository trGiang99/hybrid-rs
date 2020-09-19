import numpy as np
from numba import njit


@njit
def _shuffle(X):
    np.random.shuffle(X)
    return X


@njit
def _run_epoch(X, pu, qi, bu, bi, S, k, global_mean, n_factors, lr_pu, lr_qi, lr_bu, lr_bi, reg_pu, reg_qi, reg_bu, reg_bi):
    """Runs an epoch, updating model weights (pu, qi, bu, bi).
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
    residuals = []
    for i in range(X.shape[0]):
        user, item, rating = int(X[i, 0]), int(X[i, 1]), X[i, 2]

        # Predict current rating
        pred = global_mean + bu[user] + bi[item]

        for factor in range(n_factors):
            pred += pu[user, factor] * qi[item, factor]

        # Find items that have been rated by user u beside item i and their ratings
        # items_ratedby_u = np.array([
        #     [int(i_id), rating_u] for i_id, u_id, rating_u in zip(X[:,1], X[:,0], X[:,2])
        #         if i_id != item and u_id == user
        # ])
        # sim_ratings = items_ratedby_u[:,1]
        # items_ratedby_u = items_ratedby_u[:,0]

        items_ratedby_u = []
        sim_ratings = []
        for i_id, u_id, rating_u in zip(X[:,1], X[:,0], X[:,2]):
            if i_id != item and u_id == user:
                items_ratedby_u.append(int(i_id))
                sim_ratings.append(rating_u)
        items_ratedby_u = np.array(items_ratedby_u)
        sim_ratings = np.array(sim_ratings)


        sim = np.array([
            S[item, item_ratedby_u] if S[item, item_ratedby_u] else S[item_ratedby_u, item] for item_ratedby_u in items_ratedby_u
        ])

        # Sort similarity list in descending and get k first indices
        k_nearest_items = np.argsort(sim)[-k:]

        # Get first k items or all if number of similar items smaller than k
        sim = sim[k_nearest_items]
        sim_ratings = sim_ratings[k_nearest_items] - pred
        # sim_ratings = np.array([
        #     X[np.where((X[:,1] == j) & (X[:,0] == user))[0][0], 2] for j in items_ratedby_u[k_nearest_items]
        # ])

        # sim_ratings -= pred

        pred += np.sum(sim * sim_ratings) / (np.abs(sim).sum() + 1e-8)

        err = rating - pred
        residuals.append(err)

        # Update biases
        bu[user] += lr_bu * (err - reg_bu * bu[user])
        bi[item] += lr_bi * (err - reg_bi * bi[item])

        # Update latent factors
        for factor in range(n_factors):
            puf = pu[user, factor]
            qif = qi[item, factor]

            pu[user, factor] += lr_pu * (err * qif - reg_pu * puf)
            qi[item, factor] += lr_qi * (err * puf - reg_qi * qif)

    residuals = np.array(residuals)
    train_loss = np.square(residuals).mean()
    return pu, qi, bu, bi, train_loss
