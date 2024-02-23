"""Plot the results from the BQ Experiment."""
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tueplots import figsizes
from tueplots import bundles
from tueplots.constants.color import rgb

import twilight_colormap
import scikits.bootstrap as sci

plt.rcParams.update(bundles.aistats2023(column="half", nrows=2, ncols=2))
plt.rcParams.update(figsizes.aistats2023_half(height_to_width_ratio=1.2))
np.random.seed(6)

colors = {
    "exact": twilight_colormap.color3,
    "uniform": twilight_colormap.color4,
    "greedy": twilight_colormap.color1,
}

plt.rcParams.update(
    {"text.usetex": True, "text.latex.preamble": r"\usepackage{amsfonts}"}
)


def plot_results_subplot(name, title, d, ls, var, ax):
    """Load the results from the log files and make a plot."""
    name = name + "_d" + str(d) + "_ls" + str(ls) + "_var" + str(var)

    greedy_df = pd.read_csv("./IntegrationExperiments/logs/" + name + "_greedy.csv")
    dpp_df = pd.read_csv("./IntegrationExperiments/logs/" + name + "_dpp.csv")
    uniform_df = pd.read_csv("./IntegrationExperiments/logs/" + name + "_uniform.csv")

    ks = greedy_df["k"].to_list()

    greedy_df = greedy_df.drop("k", axis=1)
    dpp_df = dpp_df.drop("k", axis=1)
    uniform_df = uniform_df.drop("k", axis=1)

    greedy_df_mean = greedy_df.mean(axis=1)
    greedy_df_var = greedy_df.std(axis=1)
    dpp_df_mean = dpp_df.mean(axis=1)
    dpp_df_var = dpp_df.std(axis=1)
    uniform_df_mean = uniform_df.mean(axis=1)
    uniform_df_var = uniform_df.std(axis=1)

    greedy_df_cis = sci.ci(greedy_df.T, statfunction=lambda x: np.average(x, axis=0))
    uniform_df_cis = sci.ci(uniform_df.T, statfunction=lambda x: np.average(x, axis=0))
    dpp_df_cis = sci.ci(dpp_df.T, statfunction=lambda x: np.average(x, axis=0))

    ax.errorbar(
        ks,
        greedy_df_mean.values,
        np.zeros(len(ks)),
        color=colors["greedy"],
        label="greedy",
        ls="dotted",
        marker="o",
        ms=2,
    )
    ax.errorbar(
        ks,
        dpp_df_mean.values,
        np.zeros(len(ks)),
        color=colors["exact"],
        label="exact",
        ls="dotted",
        marker="o",
        ms=2,
    )
    ax.errorbar(
        ks,
        uniform_df_mean.values,
        np.zeros(len(ks)),
        color=colors["uniform"],
        label="uniform",
        ls="dotted",
        marker="o",
        ms=2,
    )

    ax.fill_between(
        ks,
        uniform_df_cis[0],
        uniform_df_cis[1],
        alpha=0.3,
        color=colors["uniform"],
    )
    ax.fill_between(
        ks,
        greedy_df_cis[0],
        greedy_df_cis[1],
        alpha=0.3,
        color=colors["greedy"],
    )
    ax.fill_between(
        ks,
        dpp_df_cis[0],
        dpp_df_cis[1],
        alpha=0.3,
        color=colors["exact"],
    )

    ax.set_xticks([2, 10, 20])
    ax.set_title(title)


#
fig, axs = plt.subplots(nrows=2, ncols=2)
plot_results_subplot(
    "continuous_integrand_family", "Continuous integrand family", 2, 0.2, 1, axs[0, 0]
)
plot_results_subplot(
    "roos_arnold_function", "Roos Arnold function", 2, 0.2, 4, axs[0, 1]
)
plot_results_subplot(
    "morokoff_caflisch_function1",
    "Morokoff Caflisch function 1",
    2,
    0.2,
    2.5,
    axs[1, 0],
)
plot_results_subplot("zhou_function", "Zhou function", 2, 0.2, 8, axs[1, 1])
axs[1, 0].set_xlabel("$k$")
axs[1, 1].set_xlabel("$k$")
axs[0, 0].set_ylabel("$\hat{\mathbb{E}}[|\hat{F}-F|]$")
axs[1, 0].set_ylabel("$\hat{\mathbb{E}}[|\hat{F}-F|]$")
handles, labels = axs[1, 1].get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(1, 0.95))#, loc="upper right")
plt.savefig("./Figures/Figure5.pdf")
plt.show()
