"""
Plot the progress of a peak over time
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from .tools import read_csvs


def plot_progress(xs, ys, times, x_point, fit=None, savefig=False, colors=None):
    """
    Plot the change of the height of a point across time
    :param inps: list of input files
    :param fit: plot a linear fit
    :param savefig: save the figure to the specified file name
    """
    # Find the height at the specified point
    heights = []
    for x_vals, y_vals in zip(xs, ys):
        for x, y in zip(x_vals, y_vals):
            if x >= x_point:
                heights.append(y)
                break

    heights = heights[1:]
    plt.plot(times, heights)
    plt.gca().set_xlabel('Time (min)')
    plt.gca().set_ylabel(f'Absorbance at {x_point} cm$^{{-1}}$')

    if colors:
        plt.scatter(times, heights, color=colors, zorder=100)

    if fit:
        slope, intercept, r_value, p_value, std_err = stats.linregress(times, heights)
        function = np.poly1d((slope, intercept))
        ends = (times[0], times[-1])
        label = f"     Fit -- R$^2$ = {r_value**2:3.2f}\n$y$ = {slope:5.5f}$X$ + {intercept:5.2f}"
        plt.plot(ends, function(ends), label=label)

    plt.gca().set_ylim(bottom=0)
    plt.legend()
    plt.suptitle(f'Peak Progress ({x_point} cm$^{{-1}}$)')

    if savefig:
        plt.savefig(savefig)
    plt.show()
