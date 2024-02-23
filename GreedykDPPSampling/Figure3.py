import numpy as np
import se_md
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tueplots import figsizes

from tueplots import bundles
from tueplots.constants.color import rgb


plt.rcParams.update(bundles.aistats2023(column="half"))
plt.rcParams.update(figsizes.aistats2023_half(height_to_width_ratio=1))


np.random.seed(6)
plt.rcParams.update(
    {"text.usetex": True, "text.latex.preamble": r"\usepackage{amsfonts}"}
)

import warnings

warnings.filterwarnings("ignore")


def sample_with_plot_2d(nb_points):

    ell = [1, 1]

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

        plot_xp = np.zeros((gridsize, gridsize))
        for ii in range(100):
            for jj in range(100):
                plot_xp[ii, jj] = myV(np.asarray([xp[ii], xp[jj]]))

        fig = plt.figure()

        gs = GridSpec(4, 4, hspace=0.6, wspace=0.6)

        ax_joint = fig.add_subplot(gs[1:4, 0:3])
        ax_joint.contour(plot_xp, cmap="twilight_r", alpha=0.3, aspect="equal")
        ax_joint.imshow(plot_xp, cmap="twilight_r", alpha=0.3, aspect="equal")

        ax_marg_x = fig.add_subplot(gs[0, 0:3])
        ax_marg_y = fig.add_subplot(gs[1:4, 3])
        ax_marg_x.margins(x=0, y=0, tight=True)
        ax_marg_y.margins(x=0, y=0, tight=True)
        ax_joint.margins(x=0, y=0, tight=True)

        if i > 0:
            Points = np.asarray(X[:-1])
            ax_joint.scatter(
                Points[:, 1] * 100, Points[:, 0] * 100, color="black", alpha=1, s=2
            )
            ax_joint.hlines(
                y=Points[:, 0] * 100,
                xmin=Points[:, 1] * 100,
                xmax=99,
                linewidth=0.5,
                color="black",
                alpha=0.5,
            )
            ax_joint.vlines(
                x=Points[:, 1] * 100,
                ymin=Points[:, 0] * 100,
                ymax=99,
                linewidth=0.5,
                color="black",
                alpha=0.5,
            )

            # And the next point
            ax_joint.scatter(
                X[-1][1] * 100, X[-1][0] * 100, color=rgb.tue_red, alpha=1, s=5
            )

            ax_joint.hlines(
                y=X[-1][0] * 100,
                xmin=X[-1][1] * 100,
                xmax=99,
                linewidth=1,
                linestyle="--",
                color=rgb.tue_red,
                alpha=1,
            )
            ax_joint.vlines(
                x=X[-1][1] * 100,
                ymin=X[-1][0] * 100,
                ymax=99,
                linewidth=1,
                linestyle="--",
                color=rgb.tue_red,
                alpha=1,
            )

            ax_marg_y.hlines(
                y=Points[:, 0],
                xmin=0,
                xmax=Z_x,
                linewidth=0.5,
                color="black",
                alpha=0.5,
            )

            ax_marg_x.vlines(
                x=Points[:, 1],
                ymin=0,
                ymax=Z_y,
                linewidth=0.5,
                color="black",
                alpha=0.5,
            )

            ax_marg_y.hlines(
                y=X[-1][0],
                xmin=0,
                xmax=u_x,
                linewidth=1,
                color=rgb.tue_red,
                linestyle="--",
                alpha=1,
            )

            ax_marg_x.vlines(
                x=X[-1][1],
                ymin=0,
                ymax=u_y,
                linewidth=1,
                color=rgb.tue_red,
                linestyle="--",
                alpha=1,
            )

        ax_joint.invert_yaxis()
        ax_marg_y.plot(cdf_x, xp, color="black", linewidth=1)
        ax_marg_x.plot(xp, cdf_y, color="black", linewidth=1)
        ax_marg_x.hlines(u_y, 0, 1, color=rgb.tue_red, linewidth=1)
        ax_marg_y.vlines(u_x, 0, 1, color=rgb.tue_red, linewidth=1)

        ax_marg_x.set_xticks([0, 1])
        ax_marg_x.set_xticklabels([0, 1])
        ax_marg_x.set_yticks([0, u_y])
        ax_marg_x.set_yticklabels([0, "$u$"])

        ax_marg_y.set_xticks([0, u_x])
        ax_marg_y.set_xticklabels([0, "$u$"])
        ax_marg_y.set_yticks([0, 1])
        ax_marg_y.set_yticklabels([0, 1])

        ax_joint.set_xticks([0, 100])
        ax_joint.set_xticklabels([0, 1])
        ax_joint.set_yticks([0, 100])
        ax_joint.set_yticklabels([0, 1])

        # Set labels on joint
        ax_joint.set_xlabel("$x_{i,2}$")
        ax_joint.set_ylabel("$x_{i,1}$")

        # Set labels on marginals
        ax_marg_y.set_xlabel("$\mathbb{V}_{i}(x_{i,1})$")
        ax_marg_x.set_ylabel("$\mathbb{V}_{i}(x_{i,2}|x_{i,1})$")
        # fig.tight_layout()
        plt.savefig("./Figures/Figure3.pdf")
        plt.show()


sample_with_plot_2d(5)
