import numpy as np

from .plot import plotter, setup_axis
from .tools import cull
from .progress import plot_spectra_progress
from .spectrum import spectra_from_csvs
from glob import glob
from datetime import datetime, timedelta

import matplotlib.pyplot as plt

from natsort import natsorted
from matplotlib import lines


def plot_reaction_kinetics(reactions, concentrations, folder, verbose=False,
                           colors=None, linestyles=None,
                           kinetics_smooth=False,
                           kinetics_x_max=60, kinetics_x_units='minutes', kinetics_y_lim=None,
                           spectra_plot=True, spectra_cull_number=8, spectra_smooth=False,
                           spectra_style='IR', spectra_xlim=None, spectra_xticks=None, spectra_xlabel=None, spectra_ylabel=None,
                           integration_x_points=(2100, 2400), baseline_region=(2500, 2600),
                           title='', combo_plot=True,
                           savefig=None):
    """
    Plot a graph of the reaction kinetics for multiple reactions.

    :param reactions: Names of the reactions (correspond to the folder)
    :param concentrations: Concentrations of the reactant of interest
    :param folder: location of the reaction folders
    :param colors: colors for the reactions
    :param linestyles: linestyles for the rounds
    :param kinetics_*: parameters for kinetics plot
    :param spectra_*: parameters for kinetics plot
    :param integration_x_points: x_points to integrate over for reaction kinetics
    :param baseline_region: region for baseline correction of spectra
    :param title: title of the plot
    :param combo_plot: plot all on a final plot
    :param savefig: (where to) save the figure
    :return: fig, axes
    """
    if colors is None:
        colors = [f'C{i}' for i in range(len(reactions))]
    elif len(colors) != len(reactions):
        raise ValueError(f'len(colors)={len(colors)} != len(reactions)={len(reactions)}')
    if linestyles is None:
        linestyles = ('-', '--', ':', '-.', (0, (4, 1, 1, 1, 1, 1)))

    # Setup figures
    fig, axes = plt.subplots(len(reactions) + int(combo_plot), 2, sharex='col', sharey='col', figsize=(10, 6))

    # Assure the proper dimension for axes
    if len(reactions) + int(combo_plot) < 2:
        axes = [axes]

    fig.subplots_adjust(hspace=0, wspace=0)
    time_divisor = {'seconds': 1, 'minutes': 60, 'hours': 60*60, 'days': 60*60*24}[kinetics_x_units]

    for i, (reaction, concentration, color, (ax1, ax2)) in enumerate(zip(reactions, concentrations, colors, axes)):
        if verbose:
            print(reaction, end=' ')

        half_lives = []
        plt.text(0.5, 0.5, reaction, horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)

        for j, linestyle in enumerate(linestyles, start=1):
            # Read in spectra
            inputs = glob(f'{folder}/{reaction}/Round {j}/*.CSV')

            if len(inputs) == 0:
                break
            if verbose:
                print('.', end='')

            # Set baseline
            # Get times and names from timestamps in input name
            # e.g. 'Mon Sep 09 10-26-50 2019 (GMT-04-00).CSV'
            strp = lambda x: datetime.strptime(x, '%a %b %d %H-%M-%S %Y')
            timestamps = [strp(inp.split('/')[-1].split(' (')[0]) for inp in inputs]

            # Sort the inputs by the timestamps
            timestamps, inputs = zip(*sorted(zip(timestamps, inputs)))

            times = [(time - timestamps[0]).total_seconds()/time_divisor for time in timestamps]

            spectra = []
            for time, inp in zip(times, inputs):
                s, *others = spectra_from_csvs(inp)
                if len(others) != 0:
                    raise ValueError(f'Multiple spectra in a CSV is not supported. File={inp}')

                if spectra_smooth:
                    s = s.smoothed(spectra_smooth)

                if baseline_region:
                    s = s.set_zero(*baseline_region)

                s.name = f'{time:.1f} {kinetics_x_units}'
                s.time = time

                spectra.append(s)

            if j == 1 and spectra_plot:
                # Only plot a subset of the spectra to avoid cluttering the figure
                s = spectra if len(spectra) < spectra_cull_number else list(cull(spectra, spectra_cull_number))
                plotter(s, baseline_subtracted=False, normalized=False, title=None,
                        plot=(fig, ax1), legend=False, smoothed=False,
                        style=None, xlim=None, xticks=None,
                        colors=None, markers=None)

                # Plot result on last graph
                plotter([spectra[-1]], baseline_subtracted=False, normalized=False, title=None,
                        plot=(fig, axes[-1][0]), legend=False, smoothed=False,
                        style=None, xlim=None, xticks=None,
                        colors=None, markers=None)

            # Scale spectra by isocyanate concentration
            scaled_spectra = [s/concentration for s in spectra]

            # Plot progess
            _, half_life = plot_spectra_progress(
                scaled_spectra,
                times,
                integration_x_points,
                x_units=kinetics_x_units,
                plot=(fig, ax2),
                label=f'{j}',
                color=color,
                linestyle=linestyle,
                smooth=kinetics_smooth,
            )

            ax2.set_xlabel(None)
            if half_life is not None:
                half_lives.append(half_life)

            if combo_plot:
                plot_spectra_progress(
                    scaled_spectra,
                    times,
                    integration_x_points,
                    x_units=kinetics_x_units,
                    plot=(fig, axes[-1][1]),
                    label=f'{reaction} - {j}',
                    color=color,
                    linestyle=linestyle,
                )

        # TODO: Perhaps 1/(Σ 1/half_life) ???
        half_life = np.average(half_lives)
        if half_life > 0:
            plt.text(0.5, 0.8, f'$t_{{1/2}} = {int(round(half_life))}$ min',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax2.transAxes)
        if verbose:
            print('\n')

    # Setup axes
    for i, (ax1, ax2) in enumerate(axes):
        setup_axis(ax1, spectra_style)
        if i != (len(axes) - 1)//2 and len(axes) > 1:
            ax1.set_ylabel(None)
            ax2.set_ylabel(None)
        ax2.yaxis.set_label_position('right')
        ax2.yaxis.set_ticks_position('right')
        ax2.legend()

    ax2.set_xlim(0, kinetics_x_max)
    ax2.set_ylim(0, kinetics_y_lim)
    if kinetics_y_lim:
        ax2.set_yticks(range(0, kinetics_y_lim, 10))

    ax2.legend([plt.Line2D([0, 1], [0, 0], color=color) for color in colors], reactions)

    if title:
        fig.suptitle(title)

    if savefig:
        if savefig is True:
            fig.savefig(f'plots/{title}.svg')
        else:
            fig.savefig(savefig)

    return fig, axes
