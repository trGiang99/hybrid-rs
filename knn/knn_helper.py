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
        n_epochs (int): number of iterations to train
        lr (float): the learning rate
        reg (float): the regularization strength
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
def _predict(x_id, y_id, y_rated, S, k, k_min):
    """Predict rating of user x for item y (if iiCF).
    Args:
        x_id (int): users Id    (if iiCF)
        y_id (int): items Id    (if iiCF)
        y_rated (ndarray): List where element `i` is ndarray of `(xs, rating)` where `xs` is all x that rated y and the ratings.
        S (ndarray): similarity matrix
        k (int): number of k-nearest neighbors
        k_min (int): number of minimum k
    Returns:
        pred (float): predicted rating of user x for item y     (if iiCF)
    """

    k_neighbors = np.zeros((k, 3))
    for y2, rating in y_rated:
        if int(y2) == y_id:
            continue       # Bo qua item dang xet
        sim = S[int(y2), y_id]
        argmin = np.argmin(k_neighbors[:, 1])
        if sim > k_neighbors[argmin, 1]:
            k_neighbors[argmin] = np.array((y2, sim, rating))

    # Compute weighted average
    sum_sim = sum_ratings = actual_k = 0
    for (nb, sim, r) in k_neighbors:
        nb = int(nb)
        if sim > 0:
            sum_sim += sim
            sum_ratings += sim * r
            actual_k += 1

    if actual_k < k_min:
        sum_ratings = 0

    if sum_sim:
        est = sum_ratings / sum_sim

    return est


@njit
def _predict_mean(x_id, y_id, y_rated, mu, S, k, k_min):
    """Predict rating of user x for item y (if iiCF).
    Args:
        x_id (int): users Id    (if iiCF)
        y_id (int): items Id    (if iiCF)
        y_rated (ndarray): List where element `i` is ndarray of `(xs, rating)` where `xs` is all x that rated y and the ratings.
        mu (ndarray): List of mean ratings of all user (if iiCF, or all item if uuCF).
        S (ndarray): similarity matrix
        k (int): number of k-nearest neighbors
        k_min (int): number of minimum k
    Returns:
        pred (float): predicted rating of user x for item y     (if iiCF)
    """

    k_neighbors = np.zeros((k, 3))
    for y2, rating in y_rated:
        if int(y2) == y_id:
            continue       # Bo qua item dang xet
        sim = S[int(y2), y_id]
        argmin = np.argmin(k_neighbors[:, 1])
        if sim > k_neighbors[argmin, 1]:
            k_neighbors[argmin] = np.array((y2, sim, rating))

    # Compute weighted average
    sum_sim = sum_ratings = actual_k = 0
    for (nb, sim, r) in k_neighbors:
        nb = int(nb)
        if sim > 0:
            sum_sim += sim
            sum_ratings += sim * (r - mu[nb])
            actual_k += 1

    if actual_k < k_min:
        sum_ratings = 0

    if sum_sim:
        est = sum_ratings / sum_sim

    return est


@njit
def _predict_baseline(x_id, y_id, y_rated, S, k, k_min, global_mean, bx, by):
    """Predict rating of user x for item y (if iiCF) using baseline estimate.
    Args:
        x_id (int): users Id    (if iiCF)
        y_id (int): items Id    (if iiCF)
        y_rated (ndarray): List where element `i` is ndarray of `(xs, rating)` where `xs` is all x that rated y and the ratings.
        X (ndarray): the training set with size (|TRAINSET|, 3)
        S (ndarray): similarity matrix
        k (int): number of k-nearest neighbors
        k_min (int): number of minimum k
        global_mean (float): mean ratings in training set
        bx (ndarray): user biases   (if iiCF)
        by (ndarray): item biases   (if iiCF)
    Returns:
        pred (float): predicted rating of user x for item y     (if iiCF)
    """

    k_neighbors = np.zeros((k, 3))
    for y2, rating in y_rated:
        if int(y2) == y_id:
            continue       # Bo qua item dang xet
        sim = S[int(y2), y_id]
        argmin = np.argmin(k_neighbors[:, 1])
        if sim > k_neighbors[argmin, 1]:
            k_neighbors[argmin] = np.array((y2, sim, rating))

    est = global_mean + bx[x_id] + by[y_id]

    # Compute weighted average
    sum_sim = sum_ratings = actual_k = 0
    for (nb, sim, r) in k_neighbors:
        nb = int(nb)
        if sim > 0:
            sum_sim += sim
            nb_bsl = global_mean + bx[x_id] + by[nb]
            sum_ratings += sim * (r - nb_bsl)
            actual_k += 1

    if actual_k < k_min:
        sum_ratings = 0

    if sum_sim:
        est += sum_ratings / sum_sim

    return est