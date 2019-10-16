import numpy as np

from .plot import plotter, setup_axis
from .tools import cull
from .progress import plot_spectra_progress
from .spectrum import spectra_from_csvs
from glob import glob
from datetime import datetime, timedelta
from itertools import zip_longest

import matplotlib.pyplot as plt

from natsort import natsorted
from matplotlib import lines


def plot_reaction_kinetics(reactions, folder,
                           norms=True,
                           verbose=False,
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
    :param folder: location of the reaction folders
    :param norms: if true, normalize start to 1. If list, normalize by values.
    :param verbose: print the reactions name and dots for each round of the reaction.
    :param colors: colors for the reactions
    :param linestyles: linestyles for the rounds
    :param kinetics_*: parameters for kinetics plot
    :param spectra_*: parameters for kinetics plot
    :param integration_x_points: x_points to integrate over for reaction kinetics
    :param baseline_region: region for baseline correction of spectra
    :param title: title of the plot
    :param combo_plot: plot all on a final plot, {True, False, 'only'}
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
    height = len(reactions) + int(combo_plot) if combo_plot is not 'only' else 1
    width = 1 + int(spectra_plot)
    fig, axes = plt.subplots(height, width, sharex='col', sharey='col', figsize=(10, 6), squeeze=False)
    axes1, axes2 = axes.T if spectra_plot else ([None]*len(axes), axes.T[0])

    fig.subplots_adjust(hspace=0, wspace=0)
    time_divisor = {'seconds': 1, 'minutes': 60, 'hours': 60*60, 'days': 60*60*24}[kinetics_x_units]

    if norms in [True, False, 'max']:
        norms = [norms]*len(reactions)

    reaction_iterator = zip_longest(reactions, norms, colors, axes1[:len(reactions)], axes2[:len(reactions)])
    for reaction, norm, color, ax1, ax2 in reaction_iterator:
        if verbose:
            print(reaction, end=' ')

        half_lives = []
        if combo_plot != 'only':
            plt.text(0.5, 0.5, reaction,
                     horizontalalignment='center', verticalalignment='center',
                     transform=ax2.transAxes)

        for i, linestyle in enumerate(linestyles, start=1):
            # Read in spectra
            inputs = glob(f'{folder}/{reaction}/Round {i}/*.CSV')

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

            if i == 1 and spectra_plot:
                # Only plot a subset of the spectra to avoid cluttering the figure
                s = spectra if len(spectra) < spectra_cull_number else list(cull(spectra, spectra_cull_number))

                if combo_plot != 'only':
                    plotter(s, baseline_subtracted=False, normalized=False, title=None,
                            plot=(fig, ax1), legend=False, smoothed=False,
                            style=None, xlim=None, xticks=None,
                            colors=None, markers=None)

                # Plot result on last graph
                if combo_plot:
                    plotter([spectra[-1]], baseline_subtracted=False, normalized=False, title=None,
                            plot=(fig, axes[-1][0]), legend=False, smoothed=False,
                            style=None, xlim=None, xticks=None,
                            colors=None, markers=None)

            # Plot progess
            half_life = None
            if combo_plot != 'only':
                _, half_life = plot_spectra_progress(
                    spectra,
                    times,
                    integration_x_points,
                    x_units=kinetics_x_units,
                    plot=(fig, ax2),
                    label=f'{i}',
                    color=color,
                    linestyle=linestyle,
                    smooth=kinetics_smooth,
                    norm=norm,
                )

            if combo_plot:
                _, half_life = plot_spectra_progress(
                    spectra,
                    times,
                    integration_x_points,
                    x_units=kinetics_x_units,
                    plot=(fig, axes[-1][-1]),
                    label=f'{reaction} - {i}',
                    color=color,
                    linestyle=linestyle,
                    smooth=kinetics_smooth,
                    norm=norm,
                )

            if half_life is not None:
                half_lives.append(half_life)

        # TODO: Perhaps 1/(Σ 1/half_life) ???
        half_life = np.average(half_lives)
        if half_life > 0 and combo_plot != 'only':
            plt.text(0.5, 0.8, f'$t_{{1/2}} = {int(round(half_life))}$ min',
                     horizontalalignment='center', verticalalignment='center',
                     transform=ax2.transAxes)
        if verbose:
            print()

    # Setup axes
    for i, (ax1, ax2) in enumerate(zip(axes1, axes2)):
        if spectra_plot:
            setup_axis(ax1, spectra_style)

        if i != (len(axes) - 1)//2 and len(axes) > 1:
            if spectra_plot:
                ax1.set_ylabel(None)
            ax2.set_ylabel(None)

        ax2.legend()
        if spectra_plot:
            ax2.yaxis.set_label_position('right')
            ax2.yaxis.set_ticks_position('right')

        if i != len(axes) - 1:
            ax2.set_xlabel(None)

    ax2.set_xlim(0, kinetics_x_max)
    ax2.set_ylim(0, kinetics_y_lim)

    ax2.legend([plt.Line2D([0, 1], [0, 0], color=color) for color in colors], reactions)

    if title:
        fig.suptitle(title)

    if savefig:
        if savefig is True:
            fig.savefig(f'plots/{title}.svg')
        else:
            fig.savefig(savefig)

    return fig, axes
