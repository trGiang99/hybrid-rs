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
def _predict(u_id, i_id, items_ratedby_u, S, k, k_min, uuCF, global_mean, bu, bi):
    """Optimize biases using SGD.
    Args:
        u (int): users Id
        i (int): items Id
        X (ndarray): utility matrix
        S (ndarray): similarity matrix
        k (int): number of k-nearest neighbors
        k_min (int): number of minimum k
        uuCF (boolean): use user-user CF or not
        global_mean (float): mean ratings in training set
        bu (ndarray): user biases
        bi (ndarray): item biases
    Returns:
        pred (float): predicted rating of user u for item i.
    """

    # k_neighbors = heapq.nlargest(k, neighbors, key=lambda t: t[1])

    k_neighbors = np.zeros((k, 3))
    for i2, rating in items_ratedby_u:
        sim = S[int(i2), i_id]
        argmin = np.argmin(k_neighbors[:, 1])
        if sim > k_neighbors[argmin, 1]:
            k_neighbors[argmin] = np.array((i2, sim, rating))

    est = global_mean + bu[u_id] + bi[i_id]

    # user_known, item_known = False, False
    # if u_id in user_list:
        # user_known = True
        # est += bu[u_id]
    # if i_id in item_list:
        # item_known = True
        # est += bi[i_id]

    # if not (user_known and item_known):
    #     return est

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

    if sum_sim:
        est += sum_ratings / sum_sim
    return est