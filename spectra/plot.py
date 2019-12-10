import numpy as np

from .tools import y_at_x
from itertools import cycle

import matplotlib
import matplotlib.pyplot as plt


def plotter(
    spectra,
    title=None, style=None,
    baseline_subtracted=False, set_zero=False, normalized=False, smoothed=False, peaks=None,
    plot=None,
    xlim=None, xticks=None, xticks_minor=True, xlabel=None,
    ylim=None, yticks=None, yticks_minor=True, ylabel=None,
    colors=None, markers=None, linestyles=None,
    legend=True,
    savefig=None
):
    """
    Plot a list of Spectra.

    :param spectra: list of spectra to plot
    :param title: title of the plot
    :param style: plot-style (e.g. IR, UV-Vis)
    :param baseline_subtracted: amount to subtract, if True, use the lowest value from each spectra
    :param set_zero: set x_value (or range of values) at which y (or y average) is set to 0.
    :param normalized: normalize all of the curves at given point (or highest if True)
    :param smoothed: number of points with which to smooth
    :param peaks: dictionary of peak picking parameters
    :param plot: (figure, axis) on which to plot, generates new figure if None
    :param x*: x-axis setup parameters
    :param y*: y-axis setup parameters
    :param colors: colors to plot the spectra
    :param markers: markers to plot the spectra
    :param linestyles: linestyles to plot the spectra
    :param legend: boolean to plot legend
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
        setup_axis(ax, style, title, xlim, xticks, xticks_minor, xlabel, ylim, yticks, yticks_minor, ylabel)

    plot_spectra(spectra, style, ax, markers=markers, linestyles=linestyles, colors=colors, peaks=peaks)

    if legend:
        ax.legend()

    if savefig:
        fig.savefig(savefig)

    return fig, ax


def plot_spectra(spectra, style, ax, colors=None, markers=None, linestyles=None, peaks=None):
    """
    Plot Spectra on an axis.

    :param spectra: the Spectra to be plotted
    :param ax: the axis on which to plot
    :param style: the plot style
    :param colors: the colors to use
    :param markers: the markers to use at each point on the plot
    :param linestyles: the styles of line to use
    :param peaks: peak highlighting parameters
    """
    colors = cycle_values(colors)
    markers = cycle_values(markers)
    linestyles = cycle_values(linestyles)

    for spectrum, color, marker, linestyle in zip(spectra, colors, markers, linestyles):
        plot_spectrum(spectrum, style, ax, color=color, marker=marker, linestyle=linestyle, peaks=peaks)


def plot_spectrum(spectrum, style, ax, color=None, marker=None, linestyle=None, peaks=None):
    """
    Plot a Spectrum on an axis

    :param spectrum: the Spectrum to be plotted
    :param ax: the axis on which to plot
    :param style: the plot style
    :param color: the color to use
    :param marker: the marker to use at each point on the plot
    :param linestyle: the style of line to use
    :param peaks: peak highlighting parameters
    """
    if style not in ['MS']:
        ax.plot(
            spectrum.xs, spectrum.ys,
            label=spectrum.name,
            color=color, marker=marker, linestyle=linestyle,
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
            'prominence': 0.1 * (spectrum.max[1] - spectrum.min[1]),
        }
        if style == 'MS':
            ms_defaults = {
                'format': '4.0f',
                'marks': None,
                'prominence': 0.05 * (spectrum.max[1] - spectrum.min[1]),
            }
            peak_defaults = {**peak_defaults, **ms_defaults}

        peaks = peak_defaults if peaks is True else {**peak_defaults, **peaks}

        peak_indices, _ = spectrum.peaks(True, prominence=peaks['prominence'])
        peak_xs, peak_ys = spectrum.xs[peak_indices], spectrum.ys[peak_indices]

        if peaks['marks']:
            ax.scatter(peak_xs, peak_ys, color=color, marker=peaks['marks'])

        if peaks['labels']:
            for x, y in zip(peak_xs, peak_ys):
                ax.text(x, y, f'{{:{peaks["format"]}}}'.format(x), verticalalignment='bottom')

        if peaks['print']:
            print('     X          Y')
            for x, y in zip(peak_xs, peak_ys):
                print(f'{x:>9.3f}  {y:>9.3f}')


def setup_axis(
    ax, style, title=None,
    xlim=None, xticks=None, xticks_minor=True, xlabel=None,
    ylim=None, yticks=None, yticks_minor=True, ylabel=None,
):
    """
    Setup the axis labels and limits.
    Autogenerates based on style for any variable set to None.

    :param ax: axis to setup
    :param style: style to use
    :param title: title of the axis
    :param *lim: limits for *-axis values
    :param *ticks: *-axis ticks
    :param *ticks_minor: *-axis minor ticks
    :param *label: label for the *-axis
    """
    # update values that are None
    up = lambda v, d: d if v is None else v
    # make ticks multiples of the tick width
    make_ticks = lambda start, end, tw: np.arange(int(start/tw)*tw, int(end/tw + 1)*tw, tw)

    backwards = False

    if style.upper() == 'IR':
        backwards = True
        xlim = up(xlim, (3500, 650))
        xticks = up(xticks, make_ticks(*xlim, -500))
        xlabel = up(xlabel, 'Energy (cm$^{-1}$)')
        ylabel = up(ylabel, 'Absorbance')

    elif style.upper() == 'UV-VIS':
        xlim = up(xlim, (200, 900))
        tw = 100
        xticks = up(xticks, make_ticks(*xlim, 100))
        xlabel = up(xlabel, 'Wavelength (nm)')
        ylabel = up(ylabel, 'Absorbance')

    elif style.upper() in ['GC', 'HPLC', 'CHROMATOGRAM']:
        xlabel = up(xlabel, 'Time (min)')
        ylabel = up(ylabel, 'Response')

    elif style.upper() == 'MS':
        xlabel = up(xlabel, 'm/z')
        ylabel = up(ylabel, 'Count')

    elif style.upper() == 'NMR':
        backwards = True
        xlim = up(xlim, (10, 0))
        xticks = up(xticks, make_ticks(*xlim, -1))
        xlabel = up(xlabel, 'ppm')

    ax.set_title(title)

    if xticks is not None:
        ax.set_xticks(xticks)
    if xticks_minor is True:
        ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    elif xticks_minor is not None:
        xticks_minor *= 1 if not backwards else -1
        ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(xticks_minor))
    if yticks is not None:
        ax.set_yticks(yticks)
    if yticks_minor is True:
        ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    elif yticks_minor is not None:
        ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(yticks_minor))

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def cycle_values(values):
    """
    Make a cycle iterator of values.
    :param values: a value or list of values to be cycled.
    :return: iterator of cycled values
    """
    if not isinstance(values, list):
        values = [values]
    return cycle(values)
