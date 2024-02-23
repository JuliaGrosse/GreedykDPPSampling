"""Posterior and unnormalized CDF for posterior for sqaured expontial kernel in one dimension."""
import math
import numpy as np
from scipy import special


def se_1d(X, ell, sigma, a=0):
    """Unnormalized 1D cumulative density function (integral of posterior variance) and
       posterior variance for GP with squared exponential kernel.

    :param np.array X: Points from previous iterations.
    :param Float ell: Kernel lengthscale.
    :param Float sigma: Noise term.
    :param Float a: Lower bound of domain. Defaults to 0.

    :return: cdf, posterior variance
    :rtype: function, function

    """

    # Case: First iteration
    if not X.size:

        def cdf(b):
            return b - a

        def V(b):
            return 1

        return cdf, V
    X = X[:, None]

    # Gram matrix (called 'K' in the code and 'L' in the paper)
    var, gamma = 1, 1 / (2 * ell ** 2)
    X_norm = np.sum(X ** 2, axis=-1)
    K = var * np.exp(-gamma * (X_norm[:, None] + X_norm[None, :] - 2 * np.dot(X, X.T)))

    # The matrix that is called 'M' in the paper
    bgamma = 1 / (4 * ell ** 2)
    B = np.exp(-bgamma * (X_norm[:, None] + X_norm[None, :] - 2 * np.dot(X, X.T)))

    # The matrix that is called little 'm' in the paper
    M = 0.5 * np.add.outer(X[:, 0], X[:, 0])

    # Add noise
    K = K + sigma ** 2 * np.eye(K.shape[0])

    # inverse of Gram matrix
    #
    # This is currently O(K^3)...Can be O(K^2) made more efficient with matrix inversion lemma.
    # However, for the range of k we used in the paper np.linalg.inv was still faster than our custom
    # implementation of the matrix inverstion lemma :(
    invK = np.linalg.inv(K)

    BtimesinvK = np.multiply(B, invK)

    # cdf
    def cdf(b):
        erf1 = special.erf((b - M) / ell)
        erf2 = special.erf(M / ell)
        cdf_at_b = (b - a) - 0.5 * np.sqrt(math.pi * ell ** 2) * np.sum(
            (erf1 + erf2) * BtimesinvK
        )
        return cdf_at_b

    # V
    def V(b):
        term1 = (b - M) / ell
        term2 = np.exp(-np.multiply(term1, term1))
        term3 = np.multiply(term2, BtimesinvK)
        V_at_b = 1 - np.sum(term3)
        return V_at_b

    return cdf, V
