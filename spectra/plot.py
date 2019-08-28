from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np

from .tools import y_at_x


def plotter(spectra,
            title=None, style=None,
            baseline_subtracted=False, set_zero=False, normalized=False, smoothed=False, peaks=None,
            plot=None, xlim=None, xticks=None,
            legend=True, colors=None, markers=None, linestyles=None,
            savefig=None
            ):
    """
    Plot a list of spectra.

    :param spectra: list of spectra to plot
    :param title: title of the plot
    :param style: plot-style (e.g. IR, UV-Vis)
    :param baseline_subtracted: amount to subtract, if True, use the lowest value from each spectra
    :param set_zero: set x_value (or range of values) at which y (or y average) is set to 0.
    :param normalized: normalize all of the curves at given point (or highest if True)
    :param smoothed: number of points with which to smooth
    :param peaks: dictionary of peak picking parameters
    :param plot: (figure, axis) on which to plot, generates new figure if None
    :param xlim: x-axis limits
    :param xticks: x-axis ticks
    :param legend: boolean to plot legend
    :param colors: colors to plot the spectra
    :param markers: markers to plot the spectra
    :param linestyles: linestyles to plot the spectra
    :param savefig: where to save the figure
    :return: figure and axes
    """
    assert all(isinstance(s, type(spectra[0])) for s in spectra[1:])

    assert not (baseline_subtracted and set_zero)

    if baseline_subtracted:
        spectra = [s.baseline_subtracted(baseline_subtracted) for s in spectra]
    elif set_zero:
        spectra = [s.set_zero(set_zero) for s in spectra]

    if normalized is not False:
        if normalized is True:
            spectra = [s / max(s.ys) for s in spectra]
        else:
            spectra = [s / y_at_x(normalized, s.xs, s.ys) for s in spectra]

    if plot is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plot

    if style is not None:
        setup_axis(ax, style, xlim, xticks)

    if title:
        fig.suptitle(title)

    for spectrum, color, marker, linestyle in zip(spectra, cycle_values(colors), cycle_values(markers), cycle_values(linestyles)):
        if style not in ['MS']:
            ax.plot(
                spectrum.xs, spectrum.ys,
                label=spectrum.name,
                marker=marker, linestyle=linestyle, color=color
            )
        else:
            ax.bar(
                spectrum.xs, spectrum.ys,
                label=spectrum.name,
                color=color
            )

        # Highlight the peaks
        if peaks:
            prominence = peaks['prominence'] if 'prominence' in peaks else 0.1
            peak_indices, _ = spectrum.peaks(True, prominence=prominence)
            xs, ys = spectrum.xs[peak_indices], spectrum.ys[peak_indices]
            ax.scatter(xs, ys, color=color, marker='x')
            # Label peaks by their location
            if 'labels' in peaks and peaks['labels'] != False:
                for x, y in zip(xs, ys):
                    ax.text(x, y, f'{x:4.1f}', verticalalignment='bottom')

    if legend:
        ax.legend()

    if savefig:
        fig.savefig(savefig)

    return fig, ax


def setup_axis(ax, style, title=None, xlim=None, xticks=None, xlabel=None, ylabel=None):
    """
    Setup the axis labels and limits. Autogenerates based on style for any variable set to None.

    :param ax: axis to setup
    :param style: style to use
    :param title: title of the axis
    :param xlim: limits for x-values
    :param xticks: x-axis ticks
    :param xlabel: label for the x-axis
    :param ylabel: label for the y-axis
    """
    # update values that are not None
    up = lambda v, d: d if v is None else v

    if style.upper() == 'IR':
        xlim = up(xlim, (3500, 650))
        xticks = up(xticks, np.arange(500, 4001, 500))
        xlabel = up(xlabel, 'Energy (cm$^{-1}$)')
        ylabel = up(ylabel, 'Absorbance')

    elif style.upper() == 'UV-VIS':
        xlim = up(xlim, (200, 900))
        xticks = up(xticks, np.arange(200, 901, 100))
        xlabel = up(xlabel, 'Wavelength (nm)')
        ylabel = up(ylabel, 'Absorbance')

    elif style.upper() == 'MS':
        xlabel = 'm/z'
        ylabel = 'Intensity'

    ax.set_title(title)
    if xticks is not None:
        ax.set_xticks(xticks)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def cycle_values(values):
    """
    Make a cycle iterator of values.
    """
    if not isinstance(values, list):
        values = [values]
    return cycle(values)
