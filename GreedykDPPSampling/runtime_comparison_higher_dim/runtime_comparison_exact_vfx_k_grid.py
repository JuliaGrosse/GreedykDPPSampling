import pstats
import cProfile as profile
import numpy as np
from dppy.finite_dpps import FiniteDPP

from sklearn.gaussian_process.kernels import RBF

from runtime_comparison.myparser import parser

args = parser.parse_args()


DIM = 3
ell = np.asarray([args.ell] * DIM)

x = np.arange(0, 1, 1/args.discretization)
y = np.arange(0, 1, 1/args.discretization)
z = np.arange(0, 1, 1/args.discretization)
X,Y,Z = np.meshgrid(x,y,z)
X=np.array([X.flatten(),Y.flatten(), Z.flatten()]).T
print("X", X)
#X = X[:, None]


Lkernel = RBF(ell)


def sample_DPP():
    DPP = FiniteDPP("likelihood", **{"L_eval_X_data": (Lkernel, X)})
    for _ in range(args.nb_samples):
        DPP.sample_exact_k_dpp(size=args.k, mode="vfx")


prof = profile.Profile()
prof.enable()
sample_DPP()
prof.disable()

filename = (
    "./runtime_comparison_higher_dim/results/exact_vfx_discretization"
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
