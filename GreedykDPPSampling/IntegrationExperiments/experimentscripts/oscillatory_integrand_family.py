"""BQ Experiment with oscillatory integrand family."""
import random
import numpy as np
import scipy

import IntegrationExperiments.benchmarkfunctions as benchmarkfunctions
import IntegrationExperiments.bayesianquadrature as bayesianquadrature

np.random.seed(28)
random.seed(28)

d = 2
ls = 0.2
var = 2
ks = list(range(2, 21, 1))
f, _ = benchmarkfunctions.oscillatory_integrand_family(d=d)

scipy_f = lambda a, b: f(np.asarray([a, b]))
result = scipy.integrate.dblquad(scipy_f, 0, 1, 0, 1)[0]

bayesianquadrature.benchmark_experiment(
    "oscillatory_integrand_family", f, result, d, ks, ls, var
)
