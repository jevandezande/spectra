import sys

sys.path.insert(0, '..')

from spectra.progress import *
from spectra.spectrum import Spectrum


def setup():
    pass


def teardown():
    pass


def test_progress():
    xs1, ys1 = np.arange(10), np.arange(1, 11)
    s1 = Spectrum('Hello World', xs1, ys1)

    xs2, ys2 = np.arange(10), np.arange(10)
    s2 = Spectrum('Hello World', xs2, ys2)

    spectra = [s1, s2]

    areas, half_life = progress(spectra, [0, 1])
    assert areas == [1.5, 0.5]
    assert half_life == 1

    areas, half_life = progress(spectra, [3, 7])
    assert areas == [24, 20]
    assert half_life == None

