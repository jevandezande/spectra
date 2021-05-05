from __future__ import annotations

from datetime import datetime
from glob import glob
from itertools import zip_longest
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .conv_spectrum import ConvSpectrum
from .plot import plotter, subplots
from .progress import plot_spectra_progress
from .tools import cull


def plot_reaction_kinetics(  # noqa: C901
    reactions: Sequence[str],
    folder: str,
    names: Sequence = None,
    title: str = "",
    verbose: bool = False,
    rounds: Iterable[int] | str = "all",
    colors: Sequence[str] = None,
    linestyles: Iterable = None,
    combo_plot: str | bool = True,
    spectra_norms: Iterable = None,
    spectra_smooth: int | bool = False,
    spectra_plot: bool = True,
    spectra_cull_number: int = 8,
    spectra_style: str = "IR",
    spectra_xlim: tuple[float, float] = None,
    spectra_xticks: tuple[float, float] = None,
    spectra_xlabel: str = None,
    spectra_ylabel: str = None,
    kinetics_norms: Iterable | str | bool = True,
    kinetics_smooth: int | bool = False,
    kinetics_xmax: float = 60,
    kinetics_x_units: str = "minutes",
    kinetics_ylim: tuple[float, float] = None,
    kinetics_dot_colors: str = None,
    baseline_region: tuple[float, float] = (2500, 2600),
    integration_x_points: tuple[float, float] = (2100, 2400),
    savefig: str = None,
):
    """
    Plot a graph of the reaction kinetics for multiple reactions.

    Note: the returned axes object is not squeezed.

    :param reactions: Names of the reactions (correspond to the folder)
    :param folder: location of the reaction folders
    :param names: Names for the reactions, if `None`, defaults to `reactions`
    :param title: title of the plot
    :param verbose: print the reactions name and dots for each round of the reaction.
    :param rounds: list of rounds to display, or 'all'
    :param colors: colors for the reactions
    :param linestyles: linestyles for the rounds
    :param combo_plot: plot all on a final plot, {True, False, 'only'}
    :param spectra_norms: arguments for spectra.normed() to run on all spectra
    :param spectra_*: parameters for kinetics plot
    :param kinetics_norms: if true, normalize start to 1. If list, normalize by values.
    :param kinetics_*: parameters for kinetics plot
    :param baseline_region: region for baseline correction of spectra
    :param integration_x_points: x_points to integrate over for reaction kinetics
    :param savefig: (where to) save the figure
    :return: fig, axes
    """
    if names is None:
        names = reactions
    if colors is None:
        colors = [f"C{i}" for i in range(len(reactions))]
    elif len(colors) != len(reactions):
        raise ValueError(f"len(colors)={len(colors)} != len(reactions)={len(reactions)}")

    if linestyles is None:
        linestyles = ("-", "--", ":", "-.", (0, (4, 1, 1, 1, 1, 1)))

    if not isinstance(rounds, str):
        ls_iter = iter(linestyles)
        linestyles = [next(ls_iter) if i + 1 in rounds else None for i in range(max(rounds))]

    # Setup figures
    height = len(reactions) + int(combo_plot) if combo_plot != "only" else 1
    width = 1 + int(spectra_plot)
    fig, axes = subplots(spectra_style, height, width, figsize=(10, 6))
    axes1, axes2 = axes.T if spectra_plot else ([None] * len(axes), axes.T[0])

    time_divisor = {
        "seconds": 1,
        "minutes": 60,
        "hours": 60 * 60,
        "days": 60 * 60 * 24,
    }[kinetics_x_units]

    if not isinstance(kinetics_norms, Iterable):
        kinetics_norms = [kinetics_norms] * len(reactions)

    reaction_iterator = zip_longest(
        reactions,
        names,
        kinetics_norms,
        colors,
        axes1[: len(reactions)],
        axes2[: len(reactions)],
    )
    for reaction, name, kinetics_norm, color, ax1, ax2 in reaction_iterator:
        if verbose:
            print(reaction, end=" ")

        half_lives = []
        if combo_plot != "only":
            plt.text(
                0.5,
                0.5,
                name,
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax2.transAxes,
            )

        for i, linestyle in enumerate(linestyles, start=1):
            if rounds != "all" and i not in rounds:
                continue

            # Read in spectra
            inputs = tuple(glob(f"{folder}/{reaction}/Round {i}/*.CSV"))

            if not inputs:
                break
            if verbose:
                print(".", end="")

            # Get times and names from timestamps in input name
            # e.g. 'Mon Sep 09 10-26-50 2019 (GMT-04-00).CSV'
            strp = lambda x: datetime.strptime(x, "%a %b %d %H-%M-%S %Y")
            times_iter = (strp(inp.split("/")[-1].split(" (")[0]) for inp in inputs)

            # Sort the inputs by the timestamps
            timestamps, inputs = zip(*sorted(zip(times_iter, inputs)))

            times = [(time - timestamps[0]).total_seconds() / time_divisor for time in timestamps]

            spectra: list[ConvSpectrum] = []
            for time, inp in zip(times, inputs):
                s, *others = ConvSpectrum.from_csvs(inp)
                assert isinstance(s, ConvSpectrum)

                if len(others) != 0:
                    raise ValueError(f"Multiple spectra in a CSV is not supported. File={inp}")

                if spectra_smooth:
                    s = s.smoothed(spectra_smooth)

                if baseline_region:
                    s = s.set_zero(*baseline_region)

                if spectra_norms:
                    s = s.normed(*spectra_norms)

                s.name = f"{name} {time:.1f} {kinetics_x_units}"
                s.time = time

                assert isinstance(s, ConvSpectrum)
                spectra.append(s)

            if i == 1 and spectra_plot:
                # Only plot a subset of the spectra to avoid cluttering the figure
                if len(spectra) < spectra_cull_number:
                    to_plot = spectra
                else:
                    to_plot = list(cull(spectra, spectra_cull_number))

                if combo_plot != "only":
                    plotter(
                        to_plot,
                        baseline_subtracted=False,
                        normalized=False,
                        title=None,
                        plot=(fig, ax1),
                        legend=False,
                        smoothed=False,
                        style=spectra_style,
                        xlim=None,
                        xticks=None,
                        colors=None,
                        markers=None,
                    )

                # Plot result on last graph
                if combo_plot:
                    plotter(
                        [spectra[-1]],
                        baseline_subtracted=False,
                        normalized=False,
                        title=None,
                        plot=(fig, axes[-1][0]),
                        legend=False,
                        smoothed=False,
                        style=spectra_style,
                        xlim=None,
                        xticks=None,
                        colors=None,
                        markers=None,
                    )

            # Plot progress
            half_life = None
            if combo_plot != "only":
                _, half_life, _, _ = plot_spectra_progress(
                    spectra,
                    times,
                    integration_x_points,
                    x_units=kinetics_x_units,
                    plot=(fig, ax2),
                    label=f"{i}",
                    color=color,
                    linestyle=linestyle,
                    smooth=kinetics_smooth,
                    norm=kinetics_norm,
                    dot_colors=kinetics_dot_colors,
                )

            if combo_plot:
                _, half_life, _, _ = plot_spectra_progress(
                    spectra,
                    times,
                    integration_x_points,
                    x_units=kinetics_x_units,
                    plot=(fig, axes[-1][-1]),
                    label=f"{name} - {i}",
                    color=color,
                    linestyle=linestyle,
                    smooth=kinetics_smooth,
                    norm=kinetics_norm,
                    dot_colors=kinetics_dot_colors,
                )

            if half_life is not None:
                half_lives.append(half_life)

        # TODO: Perhaps 1/(Σ 1/half_life) ???
        half_life = float(np.average(half_lives))

        if half_life > 0 and combo_plot != "only":
            plt.text(
                0.5,
                0.8,
                f"$t_{{1/2}} = {int(round(half_life))}$ {kinetics_x_units}",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax2.transAxes,
            )
        if verbose:
            print(f" t_1/2={half_life:6.2f} {kinetics_x_units}")

    # Setup axes
    for i, (ax1, ax2) in enumerate(zip(axes1, axes2)):
        ax2.legend()
        if spectra_plot:
            ax2.yaxis.set_label_position("right")
            ax2.yaxis.set_ticks_position("right")

    ax2.set_xlim(0, kinetics_xmax)
    ax2.set_ylim(0, kinetics_ylim)

    if combo_plot:
        ax2.legend([plt.Line2D([0, 1], [0, 0], color=color) for color in colors], names)

    if title:
        fig.suptitle(title)

    if savefig is True:
        Path("plots").mkdir(exist_ok=True)
        fig.savefig(f"plots/{title}.svg")
    elif savefig:
        fig.savefig(savefig)

    return fig, axes
