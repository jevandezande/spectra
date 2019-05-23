import numpy as np

from abc import abstractmethod
from .tools import read_csvs, smooth_curve, y_at_x

from matplotlib import pyplot as plt


class Spectrum:
    def __init__(self, name, xs, ys, units=''):
        """
        A Spectrum
        :param name: name of the spectrum
        :param xs: x-values
        :param ys: intensities
        :param units: units for the x-values
        """
        assert isinstance(xs, np.ndarray)
        assert isinstance(ys, np.ndarray)
        assert len(xs.shape) == 1
        assert xs.shape == ys.shape

        self.name = name
        self.xs = xs
        self.ys = ys
        self.units = units

    def __iter__(self):
        """
        Iterate over points
        """
        yield from zip(self.xs, self.ys)

    def __eq__(self, other):
        return self.name == other.name \
            and all(self.xs == other.xs) \
            and all(self.ys == other.ys) \
            and self.units == other.units

    def __len__(self):
        """
        Number of points
        """
        return len(self.xs)

    def __repr__(self):
        return f'<{self.__class__.__name__}: {self.name}>'

    def __str__(self):
        return repr(self)

    def __rsub__(self, other):
        return (-1*self).__add__(other)
    def __sub__(self, other):
        return self.__add__(-1*other)

    def __radd__(self, other):
        return self.__add__(other)
    def __add__(self, other):
        if isinstance(other, Spectrum):
            if type(self) != type(other):
                raise TypeError(f'Cannot add spectra of different types: {type(self)} != {type(other)}.')
            if self.xs != other.xs:
                raise NotImplementedError(f'Cannot add spectra with different x-values')
            return self.__class__(f'{self.name} + {other.name}', np.copy(self.xs), self.ys + other.ys)
        return self.__class__(f'{self.name}', np.copy(self.xs), self.ys + other)

    def __rtruediv__(self, other):
        if isinstance(other, Spectrum):
            return 1/self.xs * other
        return Spectrum(self.name, self.xs, 1/self.ys) * other
    def __truediv__(self, other):
        if isinstance(other, Spectrum):
            return 1/other.ys * self
        return 1/other*self

    def __rmul__(self, other):
        return self.__mul__(other)
    def __mul__(self, other):
        if isinstance(other, Spectrum):
            if type(self) != type(other):
                raise TypeError(f'Cannot add spectra of different types: {type(self)} != {type(other)}.')
            if self.xs != other.xs:
                raise NotImplementedError(f'Cannot add spectra with different x-values')
            return self.__class__(f'{self.name} + {other.name}', np.copy(self.xs), self.ys + other.ys)
        return self.__class__(f'{self.name}', np.copy(self.xs), self.ys * other)

    @property
    def domain(self):
        """
        Domain of the spectrum (range of x-values)
        """
        return self.xs[0], self.xs[-1]

    def smoothed(self, box_pts=True):
        """
        Generate a smoothed version of the spectrum
        :param box_pts: number of data points to convolve, if True, use 3
        :return: smoothed Spectrum
        """
        return self.__class__(self.name, np.copy(self.xs), smooth_curve(self.ys, box_pts))

    def baseline_subtracted(self, val=None):
        """
        Subtract the baseline
        :param val: amount to subtract, if none, use the lowest value
        """
        if val is None:
            val = self.ys.min()
        return self.__class__(self.name, np.copy(self.xs), self.ys - val)


def spectra_from_csvs(*inps, names=None):
    """
    Read from a csv. Must only contain two columns: xs and ys.
    :param inps: file names of the csvs
    """
    if names:
        _, x_vals, y_vals = read_csvs(inps)
    else:
        names, x_vals, y_vals = read_csvs(inps)
    return [Spectrum(name, xs, ys) for name, xs, ys in zip(names, x_vals, y_vals)]
