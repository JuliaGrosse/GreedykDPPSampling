import math
import numpy as np
from scipy import special
from sklearn.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt


def se_md(X, ell, sigma, a=0):
    """Unnormalized multi-dimensional cumulative density function (intergral of posterior variance)
       and posterior variance itself for a GP with squared exponential kernel.

    :param np.array X: Points from previous iterations.
    :param [Float] ell: Lengthscales along dimensions.
    :param Float sigma: Noise term.
    :param Float a: Lower bound of domain. Defaults to 0.
    :return: cdf, pdf of posterior variance
    :rtype: function, function

    """
    d = len(ell)

    # Case: First iteration
    if not X.size:

        def cdf(b):
            return b[-1]

        def V(b):
            return 1

        return cdf, V

    N = X.shape[0]

    # Gram matrix
    var = 1
    RBFK = RBF(np.asarray(ell))
    K = var * RBFK(X)

    # The matrix that is called "M" in the paper
    gamma = np.asarray(ell) * np.sqrt(2)
    MRBFK = RBF(gamma)
    B = MRBFK(X)

    # The matrix that is called little 'm' in the paper
    # M = 0.5 * np.einsum('nd,nd->nnd', X, X)
    M = 0.5 * np.add(X.reshape(N, 1, d), X.reshape(1, N, d))

    # Add noise
    K = K + sigma ** 2 * np.eye(K.shape[0])

    # inverse of the Gram matrix (ineffcient, but hey ...)
    invK = np.linalg.inv(K)

    BtimesinvK = np.multiply(B, invK)

    # cdf
    def cdf(x):
        k = len(x) - 1
        x = np.asarray(x)

        if k < 0:
            term1 = BtimesinvK
            assert BtimesinvK.shape == (N, N)
            ell1 = np.asarray(ell[k])
            Mdivell = np.divide(M, np.asarray(ell))
            assert Mdivell.shape == (N, N, d)

            const1 = (np.sqrt(math.pi) * ell1) / 2
            matrixterm = x / ell1 - Mdivell[..., k]
            assert matrixterm.shape == (N, N)
            term2 = (
                special.erf((x / ell1 - Mdivell[..., k])) + special.erf(Mdivell[..., k])
            ) * const1
            assert term2.shape == (N, N)

            onedivell = np.divide(np.ones(len(ell)), np.asarray(ell))
            const2 = (np.sqrt(math.pi) * np.asarray(ell)[k + 1 :]) / 2
            term3 = special.erf(
                onedivell[k + 1 :] - Mdivell[..., k + 1 :]
            ) + special.erf(Mdivell[..., k + 1 :])
            assert term3.shape == (N, N, 1)
            term4 = np.prod(np.multiply(term3, const2), axis=-1)
            assert term4.shape == (N, N)
            product_all = np.multiply(np.multiply(term1, term2), term4)
            assert product_all.shape == (N, N)
            return x[k] - np.sum(product_all)
        elif k == d - 1:
            xdivell = np.divide(x, ell)
            Mdivell = np.divide(M, np.asarray(ell))
            assert Mdivell.shape == (N, N, d)

            xminusM = xdivell[:k] - Mdivell[..., :k]
            xMs = np.exp(-np.sum(np.multiply(xminusM, xminusM), axis=-1))
            assert xMs.shape == (N, N)
            term1 = np.multiply(xMs, BtimesinvK)
            assert term1.shape == (N, N)

            const1 = (np.sqrt(math.pi) * ell[k]) / 2
            ell1 = ell[k]
            term2 = (
                special.erf((x[..., k] / ell1 - Mdivell[..., k]))
                + special.erf(Mdivell[..., k])
            ) * const1
            assert term2.shape == (N, N)
            return x[k] - np.sum(np.multiply(term1, term2))
        else:
            xdivell = np.divide(x, ell[: k + 1])
            Mdivell = np.divide(M, np.asarray(ell))
            ell1 = np.asarray(ell[k])
            assert Mdivell.shape == (N, N, d)

            xminusM = xdivell[:k] - Mdivell[..., :k]
            assert xminusM.shape == (N, N, k)
            xMs = np.exp(-np.sum(np.multiply(xminusM, xminusM), axis=-1))
            assert xMs.shape == (N, N)
            term1 = np.multiply(xMs, BtimesinvK)
            assert term1.shape == (N, N)

            const1 = (np.sqrt(math.pi) * ell[k]) / 2
            term2 = (
                special.erf((x[k] / ell1 - Mdivell[..., k]))
                + special.erf(Mdivell[..., k])
            ) * const1
            assert term2.shape == (N, N)

            onedivell = np.divide(np.ones(len(ell)), np.asarray(ell))
            const2 = (np.sqrt(math.pi) * np.asarray(ell)[k + 1 :]) / 2
            term3 = special.erf(
                onedivell[k + 1 :] - Mdivell[..., k + 1 :]
            ) + special.erf(Mdivell[..., k + 1 :])
            assert term3.shape == (N, N, d - k - 1)
            term4 = np.prod(np.multiply(term3, const2), axis=-1)
            assert term4.shape == (N, N)
            return x[k] - np.sum(np.multiply(np.multiply(term1, term2), term4))

    # V
    def V(b):
        gamma1 = np.asarray(ell) / np.sqrt(2)
        NRBFK = RBF(gamma1)
        term2 = NRBFK(b[None, :], M.reshape(N * N, d)).reshape(N * N)
        term3 = np.multiply(term2, BtimesinvK.reshape(N * N))
        V_at_b = 1 - np.sum(term3)
        return V_at_b

    return cdf, V
