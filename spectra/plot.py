import numpy as np

from itertools import cycle

import matplotlib.pyplot as plt

from .tools import y_at_x


def plotter(
    spectra,
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
        try:
            x, x2 = set_zero
        except (TypeError, ValueError):
            x, x2 = set_zero, None
        spectra = [s.set_zero(x, x2) for s in spectra]

    if normalized is True:
        spectra = [s / max(s.ys) for s in spectra]
    elif normalized is not False:
        spectra = [s / y_at_x(normalized, s.xs, s.ys) for s in spectra]

    if plot is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plot

    if style is not None:
        setup_axis(ax, style, xlim, xticks)

    if title:
        fig.suptitle(title)

    plot_spectra(spectra, style, ax, markers=markers, linestyles=linestyles, colors=colors, peaks=peaks)

    if legend:
        ax.legend()

    if savefig:
        fig.savefig(savefig)

    return fig, ax


def plot_spectra(spectra, style, ax, markers=None, linestyles=None, colors=None, peaks=None):
    """
    Plot Spectra on an axis.

    :param spectra: the Spectra to be plotted
    :param ax: the axis on which to plot
    :param style: the plot style
    :param markers: the markers to use at each point on the plot
    :param linestyles: the styles of line to use
    :param colors: the colors to use
    :param peaks: peak highlighting parameters
    """
    colors = cycle_values(colors)
    markers = cycle_values(markers)
    linestyles = cycle_values(linestyles)

    for spectrum, color, marker, linestyle in zip(spectra, colors, markers, linestyles):
        plot_spectrum(spectrum, style, ax, marker=marker, linestyle=linestyle, color=color, peaks=peaks)


def plot_spectrum(spectrum, style, ax, marker=None, linestyle=None, color=None, peaks=None):
    """
    Plot a Spectrum on an axis

    :param spectrum: the Spectrum to be plotted
    :param ax: the axis on which to plot
    :param style: the plot style
    :param marker: the marker to use at each point on the plot
    :param linestyle: the style of line to use
    :param color: the color to use
    :param peaks: peak highlighting parameters
    """
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

    if peaks:
        peak_defaults = {
            'format': '4.1f',
            'labels': True,
            'marks': 'x',
            'print': True,
            'prominence': 0.1,
        }
        peaks = peak_defaults if peaks is True else {**peak_defaults, **peaks}

        peak_indices, _ = spectrum.peaks(True, prominence=peaks['prominence'])
        xs, ys = spectrum.xs[peak_indices], spectrum.ys[peak_indices]

        if peaks['marks']:
            ax.scatter(xs, ys, color=color, marker=peaks['marks'])

        if peaks['labels']:
            for x, y in zip(xs, ys):
                ax.text(x, y, f'{{:{peaks["format"]}}}'.format(x), verticalalignment='bottom')

        if peaks['print']:
            print('     X          Y')
            for x, y in zip(xs, ys):
                print(f'{x:>9.3f}  {y:>9.3f}')


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

    elif style.upper() == 'GC':
        xlabel = up(xlabel, 'Time (nm)')
        ylabel = up(ylabel, 'Response')

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
