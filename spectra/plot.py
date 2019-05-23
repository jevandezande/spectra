from .tools import y_at_x
from itertools import cycle

import matplotlib.pyplot as plt


def plotter(spectra, baseline_subtracted=True, normalized=False, title=None,
            plot=None, legend=True, smoothed=False,
            style=None, xlim=None, xticks=None,
            colors=None, markers=None):
    """
    Plot a list of spectra
    :param spectra: list of spectra to plot
    :param baseline_subtracted: amount to subtract, if True, use the lowest value from each spectra
    :param normalized: normalize all of the curves at given point (or highest if True)
    :param title: title of the plot
    :param plot: (figure, axis) on which to plot, generates new figure if None
    :param legend: boolean to plot legend
    :param smoothed: number of points with which to smooth
    :param style: plot-style (e.g. IR, UV-Vis)
    :param xlim: x-axis limits
    :param colors: colors to plot the spectra
    :param markers: markers to plot the spectra
    :return: figure and axes
    """
    assert all(isinstance(s, type(spectra[0])) for s in spectra[1:])

    if baseline_subtracted:
        spectra = [s.baseline_subtracted() for s in spectra]

    if normalized is not False:
        if normalized is True:
            spectra = [s/max(s.ys) for s in spectra]
        else:
            spectra = [s/y_at_x(normalized, s.xs, s.ys) for s in spectra]

    if plot is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plot

    if style is not None:
        setup_axis(ax, style, xlim, xticks)

    if title:
        fig.suptitle(title)

    for spectrum, color, marker in zip(spectra, cycle_values(colors), cycle_values(markers)):
        ax.plot(
            spectrum.xs, spectrum.ys,
            label=spectrum.name,
            marker=marker, color=color
        )

    if legend:
        ax.legend()

    return fig, ax


def setup_axis(ax, style, xlim=None, xticks=None, xlabel=None, ylabel=None):
    """
    Setup the axis labels and limits. Autogenerates based on style for any variable set to None.
    :param ax: axis to setup
    :param style: style to use
    :param xlim: limits for x-values
    :param xticks: ticks to plot
    :param xlabel: label for the x-axis
    :param ylabel: label for the y-axis
    """
    # update values that are not None
    up = lambda v, d: d if v is None else v

    if style.upper() == 'IR':
        xlim = up(xlim, (3500, 700))
        xticks = up(xticks, (500, 4001, 500))
        xlabel = up(xlabel, 'Energy (cm$^{-1}$)')
        ylabel = up(ylabel, 'Absorbance')

    elif style.upper() == 'UV-VIS':
        xlim = up(xlim, (200, 900))
        xticks = up(xticks, range(200, 901, 100))
        xlabel = up(xlabel, 'Wavelength (nm)')
        ylabel = up(ylabel, 'Absorbance')

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


if __name__ == '__main__':
    import numpy as np

    from spectra.spectrum import Spectrum

    s1 = Spectrum('A', np.arange(10), np.arange(10))
    s2 = Spectrum('B', np.arange(10), 10-np.arange(10))

    spectra = [s1, s2]
    fig, ax = plotter(spectra, baseline_subtracted=None, normalize=None, title='My Plot',
            plot=None, legend=True, smooth=False, style=None, xlim=None, xticks=None,
            colors=None, markers=None)

    plt.show()