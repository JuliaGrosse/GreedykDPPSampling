"""Bayesian qudrature (BQ) with different initial designs."""
import numpy as np
import pandas as pd

from sklearn.gaussian_process.kernels import RBF
import scipy

import dppy
from dppy.finite_dpps import FiniteDPP
from dppy.utils import example_eval_L_linear
import GPy

from emukit.model_wrappers.gpy_quadrature_wrappers import BaseGaussianProcessGPy, RBFGPy
from emukit.quadrature.kernels import QuadratureRBFLebesgueMeasure
from emukit.quadrature.measures import LebesgueMeasure
from emukit.quadrature.methods import VanillaBayesianQuadrature
from emukit.quadrature.loop import VanillaBayesianQuadratureLoop

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tueplots import figsizes
from tueplots import bundles
from tueplots.constants.color import rgb

import IntegrationExperiments.benchmarkfunctions
import sample_se_md
import twilight_colormap

plt.rcParams.update(bundles.aistats2022())
np.random.seed(6)

colors = {
    "exact": twilight_colormap.color3,
    "exact_vfx": twilight_colormap.color2,
    "greedy": twilight_colormap.color1,
    "mcmc_discrete": twilight_colormap.color4,
}


def uniform_samples(N, k, d):
    """Draw uniform samples."""
    return np.random.random((N, k, d))


def dpp_samples(N, k, d):
    """Draw samples from a DPP."""
    assert d == 2
    ell = [0.2] * d
    X = np.mgrid[0:1:50j, 0:1:50j]
    X = X.reshape(d, -1).T

    Lkernel = RBF(np.asarray(ell))
    DPP = FiniteDPP("likelihood", **{"L_eval_X_data": (Lkernel, X)})
    # DPP = FiniteDPP("likelihood", **{"L": Lkernel(X)})
    DPP.flush_samples()
    for _ in range(N):
        DPP.sample_exact_k_dpp(size=k, mode="alpha")

    DPPsamples = np.asarray(DPP.list_of_samples)
    DPPsamples = X[DPPsamples, :]
    return DPPsamples


def greedy_samples(N, k, d):
    """Draw samples from the greedy approximation."""
    ell = [0.2] * d

    samples = []
    for n in range(N):
        sample = sample_se_md.sample_in_md(ell, k)
        samples.append(sample)
    return samples


def bayesian_qudrature(integrand, initial_design, lengthscale=0.2, variance=1):
    """Perform BQ of the integrand with the specified initial design."""
    Y = []
    for x in initial_design:
        Y.append([integrand(x)])
    Y = np.asarray(Y)

    gpy_model = GPy.models.GPRegression(
        X=initial_design,
        Y=Y,
        kernel=GPy.kern.RBF(
            input_dim=initial_design.shape[1],
            lengthscale=lengthscale,
            variance=variance,
        ),
    )

    emukit_rbf = RBFGPy(gpy_model.kern)
    emukit_measure = LebesgueMeasure.from_bounds(bounds=[(0, 1), (0, 1)])
    emukit_qrbf = QuadratureRBFLebesgueMeasure(emukit_rbf, emukit_measure)
    emukit_model = BaseGaussianProcessGPy(kern=emukit_qrbf, gpy_model=gpy_model)

    emukit_method = VanillaBayesianQuadrature(
        base_gp=emukit_model, X=initial_design, Y=Y
    )

    emukit_loop = VanillaBayesianQuadratureLoop(model=emukit_method)

    integral_mean, integral_variance = emukit_loop.model.integrate()
    return integral_mean


def benchmark_uniform(f, k, d, rep=100, result=1, lengthscale=0.2, variance=1):
    """Repeat BQ with uniform design multiple times."""
    samples = uniform_samples(rep, k=k, d=d)
    integral_means = []
    for sample in samples:
        integral_mean = bayesian_qudrature(f, sample, lengthscale, variance)
        integral_means.append(np.abs(integral_mean - result))
    return integral_means


def benchmark_dpp(f, k, d, rep=100, result=1, lengthscale=0.2, variance=1):
    """Repeat BQ with DPP design multiple times."""
    samples = dpp_samples(rep, k=k, d=d)
    integral_means = []
    for sample in samples:
        integral_mean = bayesian_qudrature(f, sample, lengthscale, variance)
        integral_means.append(np.abs(integral_mean - result))
    return integral_means


def benchmark_greedy(f, k, d, rep=100, result=1, lengthscale=0.2, variance=1):
    """Repeat BQ with greedy design multiple times."""
    samples = greedy_samples(rep, k=k, d=d)
    integral_means = []
    for sample in samples:
        integral_mean = bayesian_qudrature(f, sample, lengthscale, variance)
        integral_means.append(np.abs(integral_mean - result))
    return integral_means


def benchmark_experiment(name, f, result, d, ks, lengthscale, variance):
    """Run everything everywhere all at once."""
    assert d == 2
    name = name + "_d" + str(d) + "_ls" + str(lengthscale) + "_var" + str(variance)

    greedy_means, dpp_means, uniform_means = [], [], []

    for k in ks:

        print(k)

        mean_greedy = benchmark_greedy(
            f, k, d, result=result, lengthscale=lengthscale, variance=variance
        )
        mean_dpp = benchmark_dpp(
            f, k, d, result=result, lengthscale=lengthscale, variance=variance
        )
        mean_uniform = benchmark_uniform(
            f, k, d, result=result, lengthscale=lengthscale, variance=variance
        )

        greedy_means.append(mean_greedy)
        dpp_means.append(mean_dpp)
        uniform_means.append(mean_uniform)


    row_names = {}
    for i, k in enumerate(ks):
        row_names[i] = k

    df_greedy = pd.DataFrame(greedy_means)
    df_greedy.rename(index=row_names, inplace=True)
    df_greedy.index.name = "k"
    df_greedy.to_csv("./IntegrationExperiments/logs/" + name + "_greedy.csv")

    df_dpp = pd.DataFrame(dpp_means)
    df_dpp.rename(index=row_names, inplace=True)
    df_dpp.index.name = "k"
    df_dpp.to_csv("./IntegrationExperiments/logs/" + name + "_dpp.csv")

    df_uniform = pd.DataFrame(uniform_means)
    df_uniform.rename(index=row_names, inplace=True)
    df_uniform.index.name = "k"
    df_uniform.to_csv("./IntegrationExperiments/logs/" + name + "_uniform.csv")
