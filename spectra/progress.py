import numpy as np

import matplotlib.pyplot as plt

from .tools import integrate, smooth_curve


def progress(spectra, x_points):
    """
    Determine the area of a region throughout multiple spectra.

    :param spectra: list of spectra
    :param x_points: range of xs to integrate over
    :return: areas, half_life_index
    """
    areas = np.array([integrate(s.xs, s.ys, x_points) for s in spectra])
    half_life_index = None
    for i, a in enumerate(areas):
        if a < areas[0] / 2:
            half_life_index = i
            break

    return areas, half_life_index


def plot_spectra_progress(
    spectra, times, x_points,
    x_units='hours',
    plot=None,
    savefig=False, label=None,
    color=None, dot_colors=None, linestyle=None,
    allow_negative=False,
    smooth=False, norm=True
):
    """
    Plot the change of the area of a region over time.

    :param spectra: iterable of spectra
    :param times: time at which curves were taken
    :param x_points: range of xs to integrate over
    :param x_units: units for the x-values
    :param plot: (fig, ax) on which to plot; if none, they will be generated
    :param savefig: save the figure to the specified file name
    :param label: label for the curve
    :param color: color of the line
    :param dot_colors: colors of the dots on the line
    :param linestyle: matplotlib linestyle
    :param allow_negative: allow the integration to be negative (otherwise converts negative values to zero)
    :param smooth: smooth the curve
    :param norm: start the curve at 1 or normalize by value
    :return: areas, half_life, fig, ax
    """
    if plot is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plot

    # Find the height at the specified point
    areas, hli = progress(spectra, x_points)
    half_life = times[hli] - times[0] if hli is not None else None

    if not allow_negative:
        areas = np.array([a if a > 0 else 0 for a in areas])

    if norm is True:
        areas /= areas[0]
    elif norm == 'max':
        areas /= max(areas)
    elif norm:
        areas /= norm

    if smooth:
        areas = smooth_curve(areas, box_pts=smooth)

    ax.plot(times, areas, label=label, color=color, linestyle=linestyle)
    ax.set_xlabel(f'Time ({x_units})')
    ax.set_ylabel(f'Absorbance peak area\n${x_points[0]}-{x_points[1]}$ cm$^{{-1}}$')

    if dot_colors:
        ax.scatter(times, areas, color=dot_colors, zorder=100)

    ax.set_ylim(bottom=0)
    fig.suptitle(f'Peak Progress (${x_points[0]}-{x_points[1]}$ cm$^{{-1}}$)')

    if savefig:
        fig.savefig(savefig)

    return areas, half_life, fig, ax
