import pstats
import cProfile as profile
import sample_se_md

from runtime_comparison.myparser import parser

args = parser.parse_args()

DIM = 3

ell = [args.ell] * DIM
def sample_DPP():
    samples = []
    for i in range(args.nb_samples):
        sample = sample_se_md.sample_in_md_runtime(ell, args.k, args.discretization)
        samples.append(sample)


prof = profile.Profile()
prof.enable()
sample_DPP()
prof.disable()

filename = (
    "./runtime_comparison_higher_dim/results/greedy_discretization"
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
