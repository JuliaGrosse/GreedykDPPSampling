import pstats
import cProfile as profile
import numpy as np
from dppy.finite_dpps import FiniteDPP
from sklearn.gaussian_process.kernels import RBF

from runtime_comparison.myparser import parser

args = parser.parse_args()

X = np.linspace(0, 1, args.discretization)
X = X[:, None]
Lkernel = RBF(np.asarray(args.ell))


# var, gamma = 1, 1 / (2 * args.ell ** 2)
# X = np.linspace(0, 1, args.discretization)
# X = X[:, None]
# X_norm = np.sum(X ** 2, axis=-1)
# Lkernel = var * np.exp(
#     -gamma * (X_norm[:, None] + X_norm[None, :] - 2 * np.dot(X, X.T))
# )

# DPP = FiniteDPP("likelihood", **{"L": Lkernel})
# DPP = FiniteDPP("likelihood", **{"L_eval_X_data": (Lkernel, X)})


def sample_DPP():
    DPP = FiniteDPP("likelihood", **{"L": Lkernel(X)})
    for _ in range(args.nb_samples):
        DPP.sample_mcmc_k_dpp(size=args.k)


# print(DPP.list_of_samples)
prof = profile.Profile()
prof.enable()
samples = sample_DPP()
prof.disable()

filename = (
    "./runtime_comparison/results/mcmc_discrete_discretization"
    + str(args.discretization)
    + "_nbsamples"
    + str(args.nb_samples)
    + "_k"
    + str(args.k)
    + "_ell"
    + str(args.ell)
    + "_rep"
    + str(args.repetition)
    + ".prof"
)

stats = pstats.Stats(prof)
stats.dump_stats(filename)
