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


def sample_DPP():
    DPP = FiniteDPP("likelihood", **{"L_eval_X_data": (Lkernel, X)})
    for _ in range(args.nb_samples):
        DPP.sample_exact_k_dpp(size=args.k, mode="alpha")


prof = profile.Profile()
prof.enable()
sample_DPP()
prof.disable()

filename = (
    "./runtime_comparison/results/exact_discretization"
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

# stats = pstats.Stats(prof)
prof.dump_stats(filename)
