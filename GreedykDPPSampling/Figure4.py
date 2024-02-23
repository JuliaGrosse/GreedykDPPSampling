import sample_se_1d
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
import numpy as np
import dppy
from dppy.finite_dpps import FiniteDPP
from dppy.utils import example_eval_L_linear

from numpy.random import rand, randn
from scipy.linalg import qr

from tueplots import figsizes


from tueplots import bundles


plt.rcParams.update(bundles.aistats2023(column="full"))
plt.rcParams.update(figsizes.aistats2023_full(rel_width=0.5, height_to_width_ratio=1))
#


ell = 1

greedyX = []
greedyY = []
for i in range(200000):
    sample = sample_se_1d.sample(ell, 2, discretization=1000000)
    greedyX.append(sample[0])
    greedyY.append(sample[1])

var, gamma = 1, 1 / (2 * ell ** 2)
X = np.linspace(0, 1, 1000)
X = X[:, None]
X_norm = np.sum(X ** 2, axis=-1)
Lkernel = var * np.exp(
    -gamma * (X_norm[:, None] + X_norm[None, :] - 2 * np.dot(X, X.T))
)

DPP = FiniteDPP("likelihood", **{"L": Lkernel})
k = 2
DPP.flush_samples()
for _ in range(200000):
    DPP.sample_exact_k_dpp(size=2)

samples = np.asarray(DPP.list_of_samples) * 0.001


# gridspec inside gridspec
fig = plt.figure()
subfigs = fig.subfigures(1, 2, wspace=0.00)

gs = GridSpec(4, 4)

#################################################
# DPP part of the plot
#################################################

# Specify layout for greedy
ax_joint = subfigs[0].add_subplot(gs[1:4, 0:3])
ax_joint.set_aspect(aspect="equal")
ax_joint.hist2d(samples[:, 0], samples[:, 1], bins=100, cmap="twilight")


ax_marg_x = subfigs[0].add_subplot(gs[0, 0:3], sharex=ax_joint)
ax_marg_y = subfigs[0].add_subplot(gs[1:4, 3], sharey=ax_joint)

# Make histogramms for greedy
ax_joint.hist2d(samples[:, 0], samples[:, 1], bins=100, cmap="twilight")
ax_marg_x.hist(samples[:, 0], bins=100, color="black")
ax_marg_y.hist(samples[:, 1], bins=100, orientation="horizontal", color="black")

ax_joint.set_xticks([0, 1])
ax_joint.set_xticklabels([0, 1])
ax_joint.set_yticks([0, 1])
ax_joint.set_yticklabels([0, 1])

ax_marg_x.set_xticks([])
ax_marg_x.set_xticklabels([])
ax_marg_x.set_yticks([])
ax_marg_x.set_yticklabels([])

ax_marg_y.set_xticks([])
ax_marg_y.set_xticklabels([])
ax_marg_y.set_yticks([])
ax_marg_y.set_yticklabels([])

ax_marg_x.margins(x=0, y=0, tight=True)
ax_marg_y.margins(x=0, y=0, tight=True)
ax_joint.margins(x=0, y=0, tight=True)

ax_marg_x.set_ylabel("")
ax_marg_x.set_xlabel("")

#################################################
# Greeedy part of the plot
#################################################

# Specify layout for greedy
ax_joint = subfigs[1].add_subplot(gs[1:4, 0:3])
ax_joint.hist2d(greedyX, greedyY, bins=100, cmap="twilight")
ax_joint.set_aspect(aspect="equal")

ax_marg_x = subfigs[1].add_subplot(gs[0, 0:3], sharex=ax_joint)
ax_marg_y = subfigs[1].add_subplot(gs[1:4, 3], sharey=ax_joint)

# Make histogramms for greedy
ax_marg_x.hist(greedyX, bins=100, color="black")
ax_marg_y.hist(greedyY, bins=100, orientation="horizontal", color="black")


ax_joint.set_xticks([0, 1])
ax_joint.set_xticklabels([0, 1])
ax_joint.set_yticks([0, 1])
ax_joint.set_yticklabels([0, 1])

ax_marg_x.set_xticks([])
ax_marg_x.set_xticklabels([])
ax_marg_x.set_yticks([])
ax_marg_x.set_yticklabels([])

ax_marg_y.set_xticks([])
ax_marg_y.set_xticklabels([])
ax_marg_y.set_yticks([])
ax_marg_y.set_yticklabels([])

ax_marg_x.margins(x=0, y=0, tight=True)
ax_marg_y.margins(x=0, y=0, tight=True)
ax_joint.margins(x=0, y=0, tight=True)

plt.savefig("./Figures/Figure4.pdf")
plt.show()
