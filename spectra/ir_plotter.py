import numpy as np

import matplotlib.pyplot as plt
from .plotter import Plotter


class IRPlotter(Plotter):
    def __init__(self, xs, ys, names, units=r'cm$^{-1}$'):
        super().__init__(xs, ys, names, units)

    def plot(self, savefig=False, title=None):
        """
        Plot the Spectra
        :param savefig: name of file to save as (False if not to be saved)
        :param title: title of the plot
        """
        fig, axes = self.setup_subplots(nrows=1, ncols=1, sharex=False, sharey=False)
        for name, x_vals, y_vals in zip(self.names, self.xs, self.ys):
            axes.plot(x_vals, y_vals, label=name)

        plt.legend()

        if title:
            fig.suptitle(title)

        if savefig:
            fig.savefig(savefig)
        plt.show()

    def setup_subplots(self, nrows=1, ncols=1, sharex=False, sharey=False):
        """
        Setup the subplots environment
        """
        fig, axes = super().setup_subplots(nrows, ncols, sharex, sharey)
        fig.suptitle('IR Plot')
        plt.xticks(range(500, 4001, 500))
        axes.set_xlim(3500, 700)
        axes.set_ylabel('Absorbance')
        axes.set_xlabel(r'Energy (cm$^{-1}$)')

        return fig, axes
