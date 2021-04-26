import numpy as np
import pytest
from numpy.testing import assert_almost_equal as aae

from spectra import SticksSpectrum


def setup():
    pass


def teardown():
    pass


def test_init():
    energies, intensities = np.arange(10), np.arange(10)
    s1 = SticksSpectrum("Hello World", energies, intensities, units="ms", style="IR", y_shift=-5, time=9)
    aae(s1.energies, energies)
    aae(s1.intensities, intensities)
    assert s1.units == "ms"
    assert s1.style == "IR"
    assert s1.y_shift == -5
    assert s1.time == 9


def test_iter():
    energies, intensities = np.arange(10), np.arange(10)
    s1 = SticksSpectrum("Hello World", energies, intensities)
    assert all(e == i for e, i in s1)


def test_eq():
    energies, intensities = np.arange(10), np.arange(10)
    s1 = SticksSpectrum("S1", energies, intensities)
    s2 = SticksSpectrum("S1", energies, intensities)
    s3 = SticksSpectrum("S1", energies, intensities, style="MS")
    s4 = SticksSpectrum("S4", energies, intensities)
    s5 = SticksSpectrum("S5", energies, intensities, y_shift=6)

    assert s1 == s2
    assert s1 != s3
    assert s1 != s4
    assert s1 != s5


def test_len():
    energies, intensities = np.arange(10), np.arange(10)
    s1 = SticksSpectrum("S1", energies, intensities)
    s2 = SticksSpectrum("S1", energies, intensities)

    assert len(s1) == len(energies)
    assert len(s2) == len(energies)


def test_str():
    energies, intensities = np.arange(10), np.arange(10)
    s1 = SticksSpectrum("Hello World", energies, intensities)

    assert str(s1) == "<SticksSpectrum: Hello World>"


def test_add_sub():
    energies1, intensities1 = np.arange(10), np.arange(10)
    energies2, intensities2 = np.arange(20), np.arange(20)
    s1 = SticksSpectrum("Hello World", energies1, intensities1)

    s1 + s1
    s2 = 1 + s1
    s3 = s2 - 1
    s4 = 1 - s3
    s5 = s1 - s1
    s6 = s1 - s2
    s7 = SticksSpectrum("Hello Big World", energies2, intensities2)

    s1 + s7
    s1 - s7

    s = s1.copy()
    s.energies += 1
    s + s1
    s - s1

    assert s1.name == "Hello World"
    assert s2.name == "Hello World + 1"
    assert s3.name == "Hello World + 1 – 1"
    assert s4.name == "1 – Hello World + 1 – 1"
    assert s5.name == "Hello World – Hello World"
    assert s6.name == "Hello World – Hello World + 1"

    aae(s1.energies, s2.energies)
    aae(s1.energies, s3.energies)
    aae(s1.energies, s4.energies)
    aae(s3.intensities, s1.intensities)


def test_abs():
    energies, intensities1, intensities2 = np.arange(10), np.arange(10), np.arange(10)
    intensities2[5:] = -intensities2[5:]
    s1 = SticksSpectrum("S1", energies, intensities1)
    s2 = SticksSpectrum("S2", energies, intensities2)

    assert s1 != s2
    assert any(s1.intensities != s2.intensities)
    aae(s1.intensities, abs(s2).intensities)


def test_mul():
    energies, intensities = np.arange(10), np.arange(10)
    s1 = SticksSpectrum("S1", energies, intensities)

    s1 * s1


def test_div():
    energies, intensities = np.arange(10), np.arange(10)
    s1 = SticksSpectrum("S1", energies, intensities)

    div = s1 / s1
    aae(div.energies, range(10))
    aae(div.intensities, [np.nan] + [1] * 9)


def test_copy():
    energies, intensities = np.arange(1, 11), np.arange(1, 11)
    s1 = SticksSpectrum("Hello World", energies, intensities)
    s2 = s1.copy()
    assert s1 == s2
    assert id(s1) != id(s2)


def test_domain():
    energies, intensities = np.arange(10), np.arange(10)
    s1 = SticksSpectrum("Hello World", energies, intensities)

    assert s1.domain == (0, 9)


@pytest.mark.xfail(raises=NotImplementedError)
def test_smoothed():
    energies, intensities = np.arange(10), np.arange(10)
    s1 = SticksSpectrum("Hello World", energies, intensities)

    s1.smoothed()


def test_baseline_subtracted():
    energies, intensities = np.arange(1, 11), np.arange(1, 11)
    s1 = SticksSpectrum("Hello World", energies, intensities)
    s2 = s1.baseline_subtracted()
    s3 = s1.baseline_subtracted(9)

    aae(s1.intensities - 1, s2.intensities)
    aae(s1.intensities - 9, s3.intensities)


@pytest.mark.xfail(raises=NotImplementedError)
def test_set_zero():
    energies, intensities = np.arange(10), np.arange(10)
    s1 = SticksSpectrum("Hello World", energies, intensities)

    s1.set_zero(99)


def test_sliced():
    energies, intensities = np.arange(10), np.arange(10)
    s1 = SticksSpectrum("Hello World", energies, intensities)

    s1.sliced()


def test_from_csvs(tmp_path):
    test_csv = f"{tmp_path}/test.csv"
    with open(test_csv, "w") as f:
        f.write("x,A,B\n0,2,4\n1,3,5")
    SticksSpectrum.from_csvs(test_csv)
    SticksSpectrum.from_csvs("tests/files/xrd.csv")


@pytest.mark.xfail(raises=NotImplementedError)
def test_norm():
    energies, intensities = np.arange(10), np.arange(10)
    s1 = SticksSpectrum("Hello World", energies, intensities)

    s1.norm()


def test_normed():
    energies, intensities = np.arange(10), np.arange(10)
    s1 = SticksSpectrum("Hello World", energies, intensities)

    s1.normed()


@pytest.mark.xfail(raises=NotImplementedError)
def test_peaks():
    energies, intensities = np.arange(10), np.arange(10)
    s1 = SticksSpectrum("Hello World", energies, intensities)

    s1.peaks()


def test_min_max():
    s1 = SticksSpectrum.from_csvs("tests/files/spectrum1.csv")[0]

    assert min(s1) == (5, 0)
    assert max(s1) == (25, 0)
    assert s1.min == (16, -10)
    assert s1.max == (13, 21)


@pytest.mark.xfail(raises=NotImplementedError)
def test_correlation():
    energies, intensities = np.arange(10), np.arange(10)
    s1 = SticksSpectrum("Hello World", energies, intensities)

    s1.correlation(s1)


def test_convert():
    energies, intensities = np.arange(10), np.arange(10)
    s1 = SticksSpectrum("Hello World", energies, intensities)

    s1.convert(2, npoints=100)
    s1.convert(2, npoints=100, energy_lim=(-5, 50))
