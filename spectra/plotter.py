import numpy as np
from abc import abstractmethod
from matplotlib import pyplot as plt

from .tools import read_csvs, y_at_x


class Plotter:
    def __init__(self, names, xs, ys, units=''):
        """
        A plotter of spectra
        :param names: names of spectra
        :param xs, ys: x and y values of spectra (each are 2-dim)
            first dimension is the index of the spectra
            second dimension is the values of the spectra
        :param units: the units for the x-axis
        """
        assert isinstance(xs, np.ndarray)
        assert isinstance(ys, np.ndarray)
        assert len(xs.shape) == 2
        assert xs.shape == ys.shape

        self.xs = xs
        self.ys = ys
        self.names = names
        self.units = units

    def __iter__(self):
        for name, x_vals, y_vals in zip(self.names, self.xs, self.ys):
            yield name, x_vals, y_vals

    @classmethod
    def from_csv(cls, inps):
        """
        Read data from csv(s).
        :param inps: string or iterable of input or inputs
        """
        return cls(*read_csvs(inps))

    @abstractmethod
    def plot(self, savefig=False, title=None, smooth=False, plot=None):
        """
        Plot the spectra
        :param savefig: name of file to save as (False if not to be saved)
        :param title: title of the plot
        :param smooth: number of points with which to smooth
        :param plot: (figure, axis) on which to plot
        """
        pass

    def subtracted(self, idxs=None):
        """
        Subtract two spectra
        :param idxs: indices of two spectra, uses the first two if None
        """
        xs, ys, names = self.xs, self.ys, self.names

        if idxs is None:
            assert xs.shape[0] == 2
            idx_a, idx_b = 0, 1
        else:
            idx_a, idx_b = idxs

        assert idx_a != idx_b
        assert 0 <= idx_a < len(self.xs)
        assert 0 <= idx_b < len(self.xs)
        assert np.all(xs[idx_a] == xs[idx_b])

        xs = np.array([xs[idx_a], xs[idx_a], xs[idx_a]])
        ys = np.array([ys[idx_a], -ys[idx_b], ys[idx_a] - ys[idx_b]])
        names = [names[idx_a], names[idx_b], 'Subtracted']

        return self.__class__(xs, ys, names, units=str(self.units))

    def subtract_baseline(self, val=None, individual=True):
        """
        Subtract the baseline from all values
        :param val: amount to subtract, if none, use the lowest value
        :param individual: subtract each baseline individually
        """
        if val:
            self.ys -= val
        else:
            if individual:
                self.ys = (self.ys.T - self.ys.min(1)).T
            else:
                self.ys -= self.ys.min()

    def set_zero(self, x_point):
        """
        Set an x at which y should be zero, and subtract all from that y
        :param x_point: x-value at which to set y to zero
        """
        for i, (_, x_vals, y_vals) in enumerate(self):
            self.ys[i] -= y_at_x(x_point, x_vals, y_vals)

    def normalize(self, x_point=None):
        """
        Normalize all spectra based on the y-value at point x
        :param x: the point with which to normalize with respect to
                  if not set, normalize based on the height of the highest peak

        Note: This is not the most efficient way to implement this
        """
        if x_point is None:
            xs_index = np.unravel_index(np.argmax(self.ys), self.ys.shape)
            norms = self.ys[:, xs_index[1]]
        else:
            norms = []
            for x_vals, y_vals in zip(self.xs, self.ys):
                for x, y in zip(x_vals, y_vals):
                    if x >= x_point:
                        norms.append(y)
                        break

        self.ys = (self.ys.T / norms).T


def smooth_curve(ys, box_pts=True):
    """
    :param ys: points to smooth
    :param box_pts: number of data points to convolve, if True, use 3
    :return: smoothed points
    """
    if box_pts is True:
        box_pts = 3

    box = np.ones(box_pts)/box_pts
    return np.convolve(ys, box, mode='same')
