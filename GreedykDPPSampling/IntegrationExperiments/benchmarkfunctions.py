"""Integration benchmark functions from http://www.sfu.ca/~ssurjano/cont.html"""
import math
import numpy as np


def continuous_integrand_family(d):
    a = np.ones(d) * 5
    u = np.ones(d) * 0.5

    def f(x):
        return np.exp(-np.sum(np.multiply(a, np.abs(x - u))))

    domain = [(0, 1)] * d
    return f, domain


def corner_peak_integrand_family(d):
    a = np.ones(d) * 5
    u = np.ones(d) * 0.5

    def f(x):
        return (1 + np.sum(np.sum(np.multiply(x, a)))) ** (-(d + 1))

    domain = [(0, 1)] * d
    return f, domain


def discontinuous_integrand_family(d):
    a = np.ones(d) * 5
    u = np.ones(d) * 0.5

    def f(x):
        if (x[0] > u[0]) or (x[1] > u[1]):
            return 0
        return np.exp(np.sum(np.multiply(x, a)))

    domain = [(0, 1)] * d
    return f, domain


def gaussian_peak_integrand_family(d):
    a = np.ones(d) * 5
    u = np.ones(d) * 0.5

    def f(x):
        a2 = np.multiply(a, a)
        xu = x - u
        xu2 = np.multiply(xu, xu)
        return np.exp(-np.sum(np.multiply(a2, xu2)))

    domain = [(0, 1)] * d
    return f, domain


def oscillatory_integrand_family(d):
    a = np.ones(d) * 5
    u = np.ones(d) * 0.5

    def f(x):
        return np.cos(2 * math.pi * u[0] + np.sum(np.multiply(a, x)))

    domain = [(0, 1)] * d
    return f, domain


def product_peak_integrand_family(d):
    a = np.ones(d) * 5
    u = np.ones(d) * 0.5

    def f(x):
        return np.prod([(1 / (a[i] ** (-2) + (x[i] - u[i]) ** 2)) for i in range(d)])

    domain = [(0, 1)] * d
    return f, domain


def g_function(d):
    """Result is =1."""
    a = np.asarray([((i + 1) - 2) / 2 for i in range(d)])

    def f(x):
        return np.prod([(np.abs(4 * x[i] - 2) + a[i]) / (1 + a[i]) for i in range(d)])

    domain = [(0, 1)] * d
    return f, domain


def morokoff_caflisch_function1(d):
    """Result is =1."""
    term1 = (1 + 1 / d) ** d

    def f(x):
        return term1 * np.prod([x[i] ** (1 / d) for i in range(d)])

    domain = [(0, 1)] * d
    return f, domain


def morokoff_caflisch_function2(d):
    term1 = 1 / (d - 0.5) ** d

    def f(x):
        return term1 * np.prod(d - x)

    domain = [(0, 1)] * d
    return f, domain


def roos_arnold_function(d):
    """Result is = 1."""

    def f(x):
        return np.prod(np.abs(4 * x - 2))

    domain = [(0, 1)] * d
    return f, domain


def bratley_function(d):
    """Result is -(1/3)*(1-(-1/2)**d)."""

    def f(x):
        return np.sum(
            [(-1) ** (i + 1) * np.prod([x[j] for j in range(i)]) for i in range(d)]
        )

    domain = [(0, 1)] * d
    return f, domain


def zhou_function(d):
    def phi(x):
        return (2 * math.pi) ** (-d / 2) * np.exp(-0.5 * np.linalg.norm(x) ** 2)

    def f(x):
        term1 = (10 ** d) / 2
        term2 = phi(10 * (x - 1 / 3))
        term3 = phi(10 * (x - 2 / 3))
        return term1 * (term2 + term3)

    domain = [(0, 1)] * d
    return f, domain
