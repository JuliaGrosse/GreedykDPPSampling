"""Greedy samling from DPP with squared exponential kernel in multiple dimensions."""
import numpy as np
import se_md

np.random.seed(6)


def sample(ell, nb_points):
    """Greedy samling from DPP with squared exponential kernel in multiple dimensions.

    :param np.array[Float] ell: Lengthscales along the dimensions.
    :param Int nb_points: k (= size of a sample = number of points).
    :return: Sample with k points.
    :rtype: [[Float]]

    """
    X = []
    for i in range(nb_points):
        cdf, myV = se_md.se_md(
            np.asarray(X), ell, 1e-5
        )  # noise for numerical stability for Linv
        x = []
        for d in range(dim):
            cdfp, Vp = np.zeros(gridsize), np.zeros(gridsize)

            for j in range(gridsize):
                y = x + [xp[j]]
                cdfp[j] = cdf(y)

            y = x + [1]
            Z = cdf(y)
            u = Z * np.random.uniform()

            interval = np.asarray([0, 1])
            while interval[1] - interval[0] > 1e-9:
                m_int = np.mean(interval)
                y = x + [m_int]
                if cdf(y) < u:
                    interval = np.asarray([m_int, interval[1]])
                else:
                    interval = np.asarray([interval[0], m_int])
            x.append(m_int)
        X.append(x)
    return X


def sample_in_md(ell, nb_points):
    X = []
    dim = len(ell)
    for i in range(nb_points):
        cdf, myV = se_md.se_md(np.asarray(X), ell, 1e-5, 0)
        x = []
        for d in range(dim):

            y = x + [1]
            Z = cdf(y)
            u = Z * np.random.uniform()

            interval = np.asarray([0, 1])
            while interval[1] - interval[0] > 1 / 50:
                m_int = np.mean(interval)
                y = x + [m_int]
                if cdf(y) < u:
                    interval = np.asarray([m_int, interval[1]])
                else:
                    interval = np.asarray([interval[0], m_int])
            x.append(m_int)
        X.append(x)

    return np.asarray(X)


def sample_in_md_runtime(ell, nb_points, discretization):
    X = []
    dim = len(ell)
    for i in range(nb_points):
        cdf, myV = se_md.se_md(np.asarray(X), ell, 1e-5, 0)
        x = []
        for d in range(dim):

            y = x + [1]
            Z = cdf(y)
            u = Z * np.random.uniform()

            interval = np.asarray([0, 1])
            while interval[1] - interval[0] > 1 / discretization:
                m_int = np.mean(interval)
                y = x + [m_int]
                if cdf(y) < u:
                    interval = np.asarray([m_int, interval[1]])
                else:
                    interval = np.asarray([interval[0], m_int])
            x.append(m_int)
        X.append(x)

    return np.asarray(X)
