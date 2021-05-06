import numpy as np
from scipy.sparse.linalg import norm as norm_sparse
from numpy.linalg import norm as norm_np

from numba import njit


def _cosine(U, uuCF):
    """Calculate cosine simularity score between object i and object j.
    Object i and j could be item or user, but must be the same.
    Assign the similarity score to S (similarity matrix).

    Args:
        U (scipy.sparse.csr_matrix): Utility matrix
        uuCF (boolean): 1 if using user-based CF, 0 if using item-based CF

    Returns:
        S (ndarray): Similarity matrix
    """
    # User based CF
    if(uuCF):
        S = np.zeros((U.shape[0], U.shape[0]))
        users = np.unique(U.nonzero()[0])

        for uidx, u in enumerate(users):
            for v in users[(uidx+1):]:
                sum_ratings = U[u,:] * U[v,:].transpose()
                if (not sum_ratings):
                    S[u,v] = 0
                    continue

                norm2_ratings_u = norm_sparse(U[u,:], 'fro')
                norm2_ratings_v = norm_sparse(U[v,:], 'fro')

                S[u,v] = (sum_ratings / (norm2_ratings_u * norm2_ratings_v)).data[0]

    # Item based CF
    else:
        S = np.zeros((U.shape[1], U.shape[1]))
        items = np.unique(U.nonzero()[1])

        for iidx, i in enumerate(items):
            for j in items[(iidx+1):]:
                sum_ratings = U[:,i].transpose() * U[:,j]
                if (not sum_ratings):
                    S[i,j] = 0
                    continue

                norm2_ratings_i = norm_sparse(U[:,i], 'fro')
                norm2_ratings_j = norm_sparse(U[:,j], 'fro')

                S[i,j] = (sum_ratings / (norm2_ratings_i * norm2_ratings_j)).data[0]

    return S


def _pcc(U, uuCF):
    S = np.zeros((U.shape[0], U.shape[0]))
    return S


@njit
def _cosine_genome(genome):
    """Calculate cosine simularity score between each movie
    using movie genome provided by MovieLens20M dataset.

    Args:
        genome (ndarray): movie genome, where each row contains genome score for that movie.

    Returns:
        S (ndarray): Similarity matrix
    """
    S = np.zeros((genome.shape[0], genome.shape[0]))

    for uidx in range(genome.shape[0]):
        for vidx in range(genome[(uidx+1):].shape[0]):
            numerator = genome[uidx].dot(genome[vidx+uidx+1])
            if (not numerator):
                S[uidx,vidx+uidx+1] = 0
                continue

            norm2_ratings_u = norm_np(genome[uidx])
            norm2_ratings_v = norm_np(genome[vidx+uidx+1])

            S[uidx,vidx+uidx+1] = (numerator / (norm2_ratings_u * norm2_ratings_v))

    return S


def _pcc_genome(genome):
    """Calculate Pearson correlation coefficient (pcc) simularity score between each movie
    using movie genome provided by MovieLens20M dataset.

    Args:
        genome (ndarray): movie genome, where each row contains genome score for that movie.

    Returns:
        S (ndarray): Similarity matrix
    """
    # Subtract mean, to calculate Pearson similarity score
    genome -= np.mean(genome, axis=1, keepdims=True)

    return _cosine_genome(genome)