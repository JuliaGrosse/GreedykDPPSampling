"""BQ Experiment with Zhou function."""
import random
import numpy as np
import scipy

import IntegrationExperiments.benchmarkfunctions as benchmarkfunctions
import IntegrationExperiments.bayesianquadrature as bayesianquadrature

np.random.seed(28)
random.seed(28)

d = 2
ls = 0.2
var = 8
ks = list(range(2, 21, 1))
f, _ = benchmarkfunctions.zhou_function(d=d)

scipy_f = lambda a, b: f(np.asarray([a, b]))
result = scipy.integrate.dblquad(scipy_f, 0, 1, 0, 1)[0]

bayesianquadrature.benchmark_experiment("zhou_function", f, result, d, ks, ls, var)
