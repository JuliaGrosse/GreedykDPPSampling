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

plt.rcParams.update(bundles.aistats2023(column="full", nrows=4, ncols=3))
plt.rcParams.update(figsizes.aistats2023_full(height_to_width_ratio=2.5))
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
    ax.set_xlabel("$k$")
    ax.set_ylabel("$\hat{\mathbb{E}}[|\hat{F}-F|]$")
    ax.set_title(title)


#
fig, axs = plt.subplots(nrows=4, ncols=3)
plot_results_subplot("bratley_function", "Brately function", 2, 0.2, 1, axs[0, 0])
plot_results_subplot(
    "continuous_integrand_family", "Continuous integrand family", 2, 0.2, 1, axs[0, 1]
)
plot_results_subplot(
    "corner_peak_integrand_family", "Corner peak integrand family", 2, 0.2, 1, axs[0, 2]
)
plot_results_subplot(
    "discontinuous_integrand_family",
    "Discontinuous integrand family",
    2,
    0.2,
    140,
    axs[1, 0],
)
plot_results_subplot("g_function", "G function", 2, 0.2, 3.5, axs[1, 1])
plot_results_subplot(
    "gaussian_peak_integrand_family",
    "Gaussian Peak integrand family",
    2,
    0.2,
    1,
    axs[1, 2],
)
plot_results_subplot(
    "morokoff_caflisch_function1",
    "Morokoff Caflisch function 1",
    2,
    0.2,
    2.5,
    axs[2, 0],
)
plot_results_subplot(
    "morokoff_caflisch_function2", "Morokoff Caflisch function 2", 2, 0.2, 1, axs[2, 1]
)
plot_results_subplot(
    "oscillatory_integrand_family", "Oscillatory integrand family", 2, 0.2, 2, axs[2, 2]
)
plot_results_subplot(
    "product_peak_integrand_family",
    "Product peak integrand family",
    2,
    0.2,
    700,
    axs[3, 0],
)
plot_results_subplot(
    "roos_arnold_function", "Roos Arnold function", 2, 0.2, 4, axs[3, 1]
)
plot_results_subplot("zhou_function", "Zhou function", 2, 0.2, 8, axs[3, 2])
handles, labels = axs[3, 2].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center")
plt.savefig("./Figures/Figure7.pdf")
plt.show()
