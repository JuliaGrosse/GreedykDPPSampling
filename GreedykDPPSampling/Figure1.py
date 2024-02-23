import numpy as np
import se_md
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from tueplots import bundles, figsizes
from tueplots.constants.color import rgb
from sklearn.gaussian_process.kernels import RBF

import dppy
from dppy.finite_dpps import FiniteDPP
from dppy.utils import example_eval_L_linear

plt.rcParams.update(bundles.aistats2023(column="half", nrows=1, ncols=3))
plt.rcParams.update(
    figsizes.aistats2023_half(nrows=1, ncols=3, height_to_width_ratio=1)
)

np.random.seed(4)

ell = [0.1, 0.1]

###########################################
# Draw a bunch of points greedily
###########################################


def sample_with_plot_2d(nb_points):

    gridsize = 100
    xp = np.linspace(0, 1, gridsize)
    X = []

    dim = 2

    for i in range(nb_points):
        cdf, myV = se_md.se_md(np.asarray(X), ell, 0, 0)
        x = []
        for d in range(dim):
            cdfp, Vp = np.zeros(gridsize), np.zeros(gridsize)

            for j in range(gridsize):
                y = x + [xp[j]]
                cdfp[j] = cdf(y)

            y = x + [1]
            Z = cdf(y)
            u = Z * np.random.uniform()

            if d == 0:
                cdf_x = cdfp
                Z_x = Z
                u_x = u
            else:
                cdf_y = cdfp
                Z_y = Z
                u_y = u

            interval = np.asarray([0, 1])
            while interval[1] - interval[0] > 1e-9:
                m_int = np.mean(interval)
                y = x + [m_int]
                if cdf(y) < u:
                    interval = np.asarray([m_int, interval[1]])
                else:
                    interval = np.asarray([interval[0], m_int])
            x.append(m_int)
        X.append(x)

    return X


#############################################
# Draw a bunch of points uniformly at random
############################################
x_unirandom = np.random.uniform(0, 1, 100)
y_unirandom = np.random.uniform(0, 1, 100)

###########################################
# Draw a bunch of points from a k-DPP
###########################################
X, Y = np.mgrid[0:1:50j, 0:1:50j]
Xdpp = np.vstack([X.ravel(), Y.ravel()]).T
var = 1
RBFK = RBF(np.asarray(ell))
Lkernel = var * RBFK(Xdpp)
DPP = FiniteDPP("likelihood", **{"L": Lkernel})
DPP.flush_samples()
DPP.sample_exact_k_dpp(size=100)

DPPsamples = np.asarray(DPP.list_of_samples)
DPPsamples = Xdpp[DPPsamples, :]

###########################################
# Make a plot
###########################################
fig, axs = plt.subplots(nrows=1, ncols=3)
X = sample_with_plot_2d(100)
Points = np.asarray(X)
axs[0].scatter(x_unirandom, y_unirandom, color="black", alpha=1, s=0.2)
axs[2].scatter(Points[:, 1], Points[:, 0], color="black", alpha=1, s=0.2)
axs[1].scatter(DPPsamples[0, :, 1], DPPsamples[0, :, 0], color="black", alpha=1, s=0.2)


axs[0].set_xticks([])
axs[1].set_xticks([])
axs[2].set_xticks([])

axs[0].set_yticks([])
axs[1].set_yticks([])
axs[2].set_yticks([])

#axs[0].set_xlabel("$x_1$")
#axs[0].set_ylabel("$x_2$")
#axs[1].set_xlabel("$x_1$")
#axs[2].set_xlabel("$x_1$")
fig.subplots_adjust(wspace=0.05, hspace=0)
#fig.tight_layout()
plt.savefig("./Figures/Figure1.pdf")
plt.show()
