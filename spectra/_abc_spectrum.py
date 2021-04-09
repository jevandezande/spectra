from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np

from .tools import read_csvs


class Spectrum(ABC):
    @abstractmethod
    def __init__(
        self,
        name: str,
        xs: np.ndarray,
        ys: np.ndarray,
        units: str = None,
        style: str = None,
        time=None,
    ):
        assert len(xs.shape) == 1
        assert xs.shape == ys.shape

        self.name = name
        self.xs = xs
        self.ys = ys
        self.units = units
        self.style = style
        self.time = time

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"

    def __str__(self) -> str:
        return repr(self)

    @abstractmethod
    def __add__(self, other) -> Spectrum:
        pass

    @abstractmethod
    def __mul__(self, other) -> Spectrum:
        pass

    @abstractmethod
    def copy(self) -> Spectrum:
        pass

    @property
    @abstractmethod
    def min(self) -> tuple[float, float]:
        pass

    @property
    @abstractmethod
    def max(self) -> tuple[float, float]:
        pass

    @property
    @abstractmethod
    def domain(self) -> tuple[float, float]:
        pass

    @classmethod
    def spectra_from_csvs(cls: type[Spectrum], *inps: str, names: Iterable[str] = None) -> list[Spectrum]:
        """
        Read from csvs.

        :param inps: file names of the csvs
        :param names: names of the Spectra
        :return: list of Spectra
        """
        ns, energies, intensities = read_csvs(inps)
        names = ns if names is None else names
        return [cls(name, energies, intensities) for name, xs, ys in zip(names, energies, intensities)]
