import numpy as np
import se_1d
import matplotlib.pyplot as plt

from tueplots import bundles
from tueplots.constants.color import rgb
from tueplots import figsizes


plt.rcParams.update(bundles.aistats2023(column="half", nrows=1, ncols=3))
plt.rcParams.update(
    figsizes.aistats2023_half(nrows=1, ncols=2, height_to_width_ratio=0.6)
)

plt.rcParams.update(
    {"text.usetex": True, "text.latex.preamble": r"\usepackage{amsfonts}"}
)

np.random.seed(3)


def sample_with_plot(nb_points):

    ell = 0.1

    gridsize = 100
    xp = np.linspace(0, 1, gridsize)
    X = []

    for i in range(nb_points):
        cdf, myV = se_1d.se_1d(np.asarray(X), ell, 0, 0)
        cdfp, Vp = np.zeros(gridsize), np.zeros(gridsize)
        for j in range(gridsize):
            cdfp[j] = cdf(xp[j])
            Vp[j] = myV(xp[j])

        Z = cdf(1)
        u = Z * np.random.uniform()

        interval = np.asarray([0, 1])
        while interval[1] - interval[0] > 1e-9:
            m_int = np.mean(interval)
            if cdf(m_int) < u:
                interval = np.asarray([m_int, interval[1]])
            else:
                interval = np.asarray([interval[0], m_int])
        X.append(m_int)

        fig, ax = plt.subplots(nrows=1, ncols=2)

        # Content
        ax[0].plot(xp, Vp, color="black", linewidth=1)
        ax[1].plot(xp, cdfp, color="black", linewidth=1)
        ax[0].scatter(
            X[:-1], np.zeros(len(X) - 1), color="black", s=7, zorder=10, clip_on=False
        )
        ax[1].scatter(
            X[:-1], np.zeros(len(X) - 1), color="black", s=7, zorder=10, clip_on=False
        )
        ax[0].scatter(X[-1], 0, color=rgb.tue_red, s=7, zorder=10, clip_on=False)
        ax[1].scatter(X[-1], 0, color=rgb.tue_red, s=7, zorder=10, clip_on=False)

        ax[1].scatter(X[-1], u, color=rgb.tue_red, s=7, zorder=10, clip_on=False)
        ax[1].scatter(0, u, color=rgb.tue_red, s=7, zorder=10, clip_on=False)

        ax[1].hlines(
            u, 0, X[-1], color=rgb.tue_red, linestyle="--", clip_on=False, linewidth=1
        )
        ax[1].vlines(
            X[-1], 0, u, color=rgb.tue_red, linestyle="--", clip_on=False, linewidth=1
        )
        ax[1].scatter(0, Z + 0.05, color="black", s=0.1, alpha=0, clip_on=False)
        ax[1].hlines(
            Z, 0, 1, color=rgb.tue_red, linestyle="-", clip_on=False, linewidth=1
        )

        # Tight layout
        ax[0].margins(x=0, y=0, tight=True)
        ax[1].margins(x=0, y=0, tight=True)

        # Ticks
        ax[0].set_yticks([0, 1])
        ax[0].set_yticklabels([0, 1])
        ax[0].set_xticks([0, 1])
        ax[0].set_xticklabels([0, 1])

        ax[1].set_yticks([0, u, Z])
        ax[1].set_yticklabels([0, "u", "Z"])
        ax[1].set_xticks([0, 1])
        ax[1].set_xticklabels([0, 1])

        # Labels
        ax[0].set_xlabel("$x$")
        ax[0].set_ylabel("$v_i(x)$")
        ax[1].set_xlabel("$x$")
        ax[1].set_ylabel("$\mathbb{V}_i(x)$")

        # plt.legend()
        plt.savefig("./Figures/Figure2.pdf")
        plt.show()


sample_with_plot(5)
