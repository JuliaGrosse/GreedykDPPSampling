import numpy as np
import se_1d

np.random.seed(3)


def sample(ell, nb_points, discretization):
    """Greedy samling from DPP with squared exponential kernel in one dimension.

    :param np.array[Float] ell: Lengthscales along the dimensions.
    :param Int nb_points: k (= size of a sample = number of points).
    :param Int discretization: N (= size of the grid)
    :return: Sample with k points.
    :rtype: [[Float]]

    """
    X = []
    for i in range(nb_points):
        cdf, myV = se_1d.se_1d(
            np.asarray(X), ell, 1e-5
        )  # some noise for numerical stability
        Z = cdf(1)
        u = Z * np.random.uniform()
        interval = np.asarray([0, 1])
        while interval[1] - interval[0] > 1 / discretization:
            m_int = np.mean(interval)
            if cdf(m_int) < u:
                interval = np.asarray([m_int, interval[1]])
            else:
                interval = np.asarray([interval[0], m_int])
        X.append(m_int)
    return X
