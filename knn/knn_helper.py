import numpy as np
from numba import njit


@njit
def _baseline_sgd(X, global_mean, n_users, n_items, n_epochs=20, lr=0.005, reg=0.02):
    """Optimize biases using SGD.
    Args:
        X (ndarray): the training set with size (|TRAINSET|, 3)
        global_mean (float): mean ratings in training set
        n_users (np.array): number of users
        n_items (np.array): number of items
    Returns:
        A tuple ``(bu, bi)``, which are users and items baselines.
    """

    bu = np.zeros(n_users)
    bi = np.zeros(n_items)

    for dummy in range(n_epochs):
        for i in range(X.shape[0]):
            user, item, rating = int(X[i, 0]), int(X[i, 1]), X[i, 2]
            err = (rating - (global_mean + bu[user] + bi[item]))
            bu[user] += lr * (err - reg * bu[user])
            bi[item] += lr * (err - reg * bi[item])

    return bu, bi


@njit
def _predict(u_id, i_id, k_neighbors, k_min, uuCF, global_mean, bu, bi):
    """Predict the fine-tune estimate of user u_id for item i_id in kNN.
    Args:
        u_id (int): users Id
        i_id (int): items Id
        k_neighbors (ndarray): k-nearest neighbors in the form of (id, sim, rating).
        k_min (int): number of minimum k
        uuCF (boolean): use user-user CF or not
        global_mean (float): mean ratings in training set
        bu (ndarray): user biases
        bi (ndarray): item biases
    Returns:
        pred (float): predicted rating of user u for item i.
    """
    # Compute weighted average
    sum_sim = sum_ratings = actual_k = 0
    for (nb, sim, r) in k_neighbors:
        nb = int(nb)
        if sim > 0:
            sum_sim += sim
            if uuCF:
                nb_bsl = global_mean + bu[nb] + bi[i_id]
            else:
                nb_bsl = global_mean + bu[u_id] + bi[nb]
            sum_ratings += sim * (r - nb_bsl)
            actual_k += 1

    if actual_k < k_min:
        sum_ratings = 0

    est = sum_ratings / (sum_sim + 10e-6)

    return est