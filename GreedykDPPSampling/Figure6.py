import matplotlib.pyplot as plt

from tueplots import bundles
from tueplots.constants.color import rgb
from tueplots import figsizes
import twilight_colormap

import pstats
import cProfile as profile

import numpy as np
import os.path

plt.rcParams.update(bundles.aistats2023(column="half", nrows=1, ncols=2))
plt.rcParams.update(figsizes.aistats2023_half(height_to_width_ratio=0.6))

plt.rcParams.update(
    {"text.usetex": True, "text.latex.preamble": r"\usepackage{amsfonts}"}
)


np.random.seed(3)

fig, ax = plt.subplots(1, 2)

algorithms = ["mcmc_discrete", "exact", "exact_vfx", "greedy"]
discretizations = [100, 1000, 10000, 50000, 60000, 70000, 100000]
ks = [10, 100]
ells = [0.01, 0.001]
reps = range(1, 6)

colors = {
    "exact": twilight_colormap.color3,
    "exact_vfx": twilight_colormap.color2,
    "greedy": twilight_colormap.color1,
    "mcmc_discrete": twilight_colormap.color4,
}
labels = {
    "exact": "alpha-DPP",
    "exact_vfx": "vfx",
    "greedy": "greedy",
    "mcmc_discrete": "mcmc",
}


def get_filename(algorithm, discretization, k, ell, rep, nb_samples=100):
    filename = (
        "./runtime_comparison/results/"
        + algorithm
        + "_discretization"
        + str(discretization)
        + "_nbsamples"
        + str(nb_samples)
        + "_k"
        + str(k)
        + "_ell"
        + str(ell)
        + "_rep"
        + str(rep)
        + ".prof"
    )
    return filename


def add_scatter(discretization, time, algorithm, k):
    if k == 10:
        a = 0
    else:
        a = 1
    if algorithm == "greedy":
        marker = "o"
    else:
        marker = "x"
    ax[a].scatter(
        discretization,
        time,
        color=colors[algorithm],
        label=labels[algorithm],
        marker=marker,
        linewidth=1,
    )


def add_error(discretization, time, error, algorithm, k):
    if k == 10:
        a = 0
    else:
        a = 1
    ax[a].errorbar(
        discretization,
        mean_time,
        yerr=error,
        color=colors[algorithm],
        label=labels[algorithm],
    )


def add_plot(discretizations, times, algorithm, k, color=None):
    if k == 10:
        a = 0
    else:
        a = 1
    if color is None:
        color = colors[algorithm]
    if algorithm == "greedy":
        marker = "o"
    else:
        marker = "x"

    ax[a].plot(
        results_discretization,
        results_time,
        color=color,
        label=labels[algorithm],
        marker=marker,
        linewidth=1,
        linestyle="-",
    )


for algorithm in algorithms:
    for k in ks:
        for ell in ells:
            results = []
            for discretization in discretizations:
                if not (k >= discretization):
                    filename = get_filename(algorithm, discretization, k, ell, 1)
                    if os.path.isfile(filename):
                        total_times = []
                        for rep in reps:
                            filename = get_filename(
                                algorithm, discretization, k, ell, rep
                            )
                            if os.path.isfile(filename):
                                stored_stats = pstats.Stats(filename)
                                total_time = stored_stats.get_stats_profile().total_tt
                                total_times.append(total_time)

                        mean_time = np.mean(total_times)
                        std_time = np.std(total_times)
                        results.append([discretization, mean_time])
                        add_scatter(discretization, mean_time, algorithm, k)
                        add_error(discretization, mean_time, std_time, algorithm, k)
            if results:
                results = np.asarray(results)
                print("results", results)
                results_discretization = results[:, 0]
                results_time = results[:, 1]
                add_plot(results_discretization, results_time, algorithm, k)
            # plt.legend()

####################################################################################################
# Add MCMC with mixing time
####################################################################################################
algorithm = "mcmc_discrete"
for k in ks:
    for ell in ells:
        results = []
        for discretization in discretizations:
            filename = get_filename(
                algorithm, discretization, k, ell, 1, nb_samples=k * discretization + k
            )
            if os.path.isfile(filename):
                total_times = []
                for rep in reps:
                    filename = get_filename(
                        algorithm,
                        discretization,
                        k,
                        ell,
                        rep,
                        nb_samples=k * discretization + k,
                    )
                    if os.path.isfile(filename):
                        stored_stats = pstats.Stats(filename)
                        total_time = stored_stats.get_stats_profile().total_tt
                        total_times.append(total_time)

                mean_time = np.mean(total_times)
                std_time = np.std(total_times)
                results.append([discretization, mean_time])
                add_scatter(discretization, mean_time, algorithm, k)
                add_error(discretization, mean_time, std_time, algorithm, k)

        if results:
            results = np.asarray(results)
            results_discretization = results[:, 0]
            results_time = results[:, 1]
            add_plot(
                results_discretization,
                results_time,
                algorithm,
                k,
                twilight_colormap.color5,
            )

ax[0].set_yscale("log")
ax[0].set_xscale("log")
ax[1].set_yscale("log")
ax[1].set_xscale("log")
ax[0].set_title("$k=10$")
ax[1].set_title("$k=100$")
ax[0].set_xlabel("discretization")
ax[1].set_xlabel("discretization")
ax[0].set_ylabel("time in s")
# ax[1].set_ylabel("time in s")

import matplotlib.patches as mpatches

handles = []
for algorithm in algorithms:
    patch = mpatches.Patch(color=colors[algorithm], label=labels[algorithm])
    handles.append(patch)
fig.subplots_adjust(bottom=0.3, wspace=0.33)
plt.legend(handles=handles, loc="upper center", bbox_to_anchor=(-0.25, -0.3), ncol=4)
#plt.legend(handles=handles)
#plt.tight_layout()
plt.savefig("./Figures/Figure6.pdf")
plt.show()
