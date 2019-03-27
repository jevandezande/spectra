import numpy as np
import matplotlib.pyplot as plt
from .plotter import Plotter, smooth_curve


class IRPlotter(Plotter):
    def __init__(self, names, xs, ys, units=r'cm$^{-1}$'):
        super().__init__(names, xs, ys, units)

    def plot(self, savefig=False, title=None, smooth=False, plot=None, legend=True):
        """
        Plot the spectra
        :param savefig: name of file to save as (False if not to be saved)
        :param title: title of the plot
        :param smooth: number of points with which to smooth
        :param plot: (figure, axis) on which to plot
        """
        if plot is None:
            fig, ax = plt.subplots()
            fig.suptitle('IR Plot')
            IRPlotter.setup_axis(ax, self.units)
        else:
            fig, ax = plot

        for name, x_vals, y_vals in zip(self.names, self.xs, self.ys):
            if smooth:
                y_vals = smooth_curve(y_vals, smooth)

            ax.plot(x_vals, y_vals, label=name)

        if legend:
            ax.legend()

        if title:
            fig.suptitle(title)

        if savefig:
            fig.savefig(savefig)

        return fig, ax

    @staticmethod
    def setup_axis(ax, units=r'cm$^{-1}$'):
        """
        Setup the axis labels and limits
        :param units: units for the x-axis
        """
        ax.set_xticks(range(500, 4001, 500))
        ax.set_xlim(3500, 700)
        ax.set_xlabel(f'Energy ({units})')
        ax.set_ylabel('Absorbance')
