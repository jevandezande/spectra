"""
Plot the progress of a peak over time
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from .tools import integrate, smooth_curve


def progress(spectra, x_points):
    """
    Determine the progress of a peak throughout multiple spectra
    :param spectra: list of spectra
    :param x_points: range of xs to integrate over
    :return: areas, half_life_index
    """
    areas = [integrate(s.xs, s.ys, x_points) for s in spectra]
    half_life_index = None
    for i, a in enumerate(areas):
        if a < areas[0]/2:
            half_life_index = i
            break

    return areas, half_life_index


def plot_spectra_progress(spectra, times, x_points, x_units='hours', fit=None, savefig=False, color=None, dot_colors=None, linestyle=None, plot=None, allow_negative=False, smooth=False, label=None):
    """
    Plot the change of the area of a point across time
    :param spectra: iterable of spectra
    :param times: time at which curves were taken
    :param x_points: range of xs to integrate over
    :param x_units: units for the x-values
    :param fit: plot a linear fit
    :param savefig: save the figure to the specified file name
    :param color: color of the line
    :param dot_colors: colors of the dots on the line
    :param linestyle: matplotlib linestyle
    :param plot: (fig, ax) on which to plot; if none, they will be generated
    :param allow_negative: allow the integration to be negative (otherwise converts negative values to zero)
    :param smooth: smooth the curve
    :param label: label for the curve
    :return: areas, half_life
    """
    if plot is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plot

    # Find the height at the specified point
    areas, hli = progress(spectra, x_points)
    half_life = times[hli] - times[0] if hli is not None else None

    if not allow_negative:
        areas = [a if a > 0 else 0 for a in areas]

    if smooth:
        areas = smooth_curve(areas, box_pts=smooth)

    ax.plot(times, areas, label=label, color=color, linestyle=linestyle)
    ax.set_xlabel(f'Time ({x_units})')
    ax.set_ylabel(f'Absorbance peak area\n${x_points[0]}-{x_points[1]}$ cm$^{{-1}}$')

    if dot_colors:
        ax.scatter(times, areas, color=dot_colors, zorder=100)

    if fit:
        slope, intercept, r_value, p_value, std_err = stats.linregress(times, areas)
        function = np.poly1d((slope, intercept))
        ends = (times[0], times[-1])
        label = f"     Fit -- R$^2$ = {r_value**2:3.2f}\n$y$ = {slope:5.5f}$X$ + {intercept:5.2f}"
        ax.plot(ends, function(ends), label=label)

    ax.set_ylim(bottom=0)
    #ax.legend()
    fig.suptitle(f'Peak Progress (${x_points[0]}-{x_points[1]}$ cm$^{{-1}}$)')

    if savefig:
        fig.savefig(savefig)

    return areas, half_life
