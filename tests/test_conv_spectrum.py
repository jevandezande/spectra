import numpy as np
from numpy.testing import assert_almost_equal as aae
from pytest import raises

from spectra.conv_spectrum import ConvSpectrum


def setup():
    pass


def teardown():
    pass


def test_init():
    energies, intensities = np.arange(10), np.arange(10)
    s1 = ConvSpectrum("Hello World", energies, intensities, units="ms", style="IR")
    aae(s1.energies, energies)
    aae(s1.intensities, intensities)
    assert s1.units == "ms"
    assert s1.style == "IR"


def test_iter():
    energies, intensities = np.arange(10), np.arange(10)
    s1 = ConvSpectrum("Hello World", energies, intensities)

    assert all(x == y for x, y in s1)


def test_eq():
    energies, intensities = np.arange(10), np.arange(10)
    s1 = ConvSpectrum("S1", energies, intensities)
    s2 = ConvSpectrum("S1", energies, intensities)
    s3 = ConvSpectrum("S3", energies, intensities)
    s4 = ConvSpectrum("S4", energies, energies)
    s5 = ConvSpectrum("S5", intensities, intensities)

    assert s1 == s2
    assert s1 != s3
    assert s1 != s4
    assert s1 != s5


def test_len():
    energies, intensities = np.arange(10), np.arange(10)
    s1 = ConvSpectrum("Hello World", energies, intensities)
    s2 = ConvSpectrum("Hello World", intensities, energies)

    assert len(s1) == len(energies)
    assert len(s2) == len(energies)


def test_str():
    energies, intensities = np.arange(10), np.arange(10)
    s1 = ConvSpectrum("Hello World", energies, intensities)

    assert str(s1) == "<ConvSpectrum: Hello World>"


def test_add_sub():
    energies1, intensities1 = np.arange(10), np.arange(10)
    energies2, intensities2 = np.arange(20), np.arange(20)
    s1 = ConvSpectrum("Hello World", energies1, intensities1)
    s1 + s1
    s2 = 1 + s1
    s3 = s2 - 1
    s4 = 1 - s3
    s5 = s1 - s1
    s6 = s1 - s2
    s7 = ConvSpectrum("Hello Big World", energies2, intensities2)

    with raises(NotImplementedError):
        s = s1.copy()
        s.energies += 1
        s + s1

    with raises(NotImplementedError):
        s = s1.copy()
        s.energies += 1
        s - s1

    with raises(NotImplementedError):
        s1 + s7

    with raises(NotImplementedError):
        s1 - s7

    assert s1.name == "Hello World"
    assert s2.name == "Hello World + 1"
    assert s3.name == "Hello World + 1 – 1"
    assert s4.name == "1 – Hello World + 1 – 1"
    assert s5.name == "Hello World – Hello World"
    assert s6.name == "Hello World – Hello World + 1"

    aae(s1.energies, s2.energies)
    aae(s1.energies, s3.energies)
    aae(s1.energies, s4.energies)
    aae(s2.intensities, np.arange(1, 11))
    aae(s3.intensities, s1.intensities)
    aae(s4.intensities, 1 - s1.intensities)
    aae(s5.intensities, 0)
    aae(s6.intensities, -1)


def test_abs():
    energies, intensities, intensities2 = np.arange(10), np.arange(10), np.arange(10)
    intensities2[5:] = -intensities2[5:]
    s1 = ConvSpectrum("Hello World", energies, intensities)
    s2 = ConvSpectrum("Hello World", energies, intensities2)

    assert s1 != s2
    assert any(s1.intensities != s2.intensities)
    aae(s1.intensities, abs(s2).intensities)


def test_mul():
    energies1, intensities1 = np.arange(10), np.arange(10)
    energies2, intensities2 = np.arange(20), np.arange(20)
    s1 = ConvSpectrum("Hello World", energies1, intensities1)
    s2 = 2 * s1
    s3 = s2 * 0.5
    s4 = s1 * s1
    s5 = ConvSpectrum("Hello Big World", energies2, intensities2)

    with raises(NotImplementedError):
        s = s1.copy()
        s.energies += 1
        s * s1

    with raises(NotImplementedError):
        s1 * s5

    aae(s1.energies, s2.energies)
    aae((s1 + s1).energies, s2.energies)
    aae((s1 + s1).intensities, s2.intensities)
    aae(s1.energies, s3.energies)
    aae(s2.intensities, np.arange(0, 20, 2))
    aae(s3.intensities, s1.intensities)
    aae(s4.intensities, s1.intensities ** 2)


def test_div():
    energies1, intensities1 = np.arange(1, 11), np.arange(1, 11)
    energies2, intensities2 = np.arange(1, 21), np.arange(1, 21)
    s1 = ConvSpectrum("Hello World", energies1, intensities1)
    s2 = 1 / s1
    s3 = s1 / 2
    s4 = s1 / s1
    s5 = ConvSpectrum("Hello World", energies1, np.array([1] * 10))
    s6 = s1 / s2
    s7 = ConvSpectrum("Hello Big World", energies2, intensities2)

    with raises(NotImplementedError):
        s = s1.copy()
        s.energies += 1
        s / s1

    with raises(NotImplementedError):
        s1 / s7

    aae(s1.energies, s2.energies)
    aae(s1.energies, s3.energies)
    aae(s2.intensities, 1 / s1.intensities)
    aae(s3.intensities, np.arange(1, 11) / 2)
    aae(s4.intensities, s5.intensities)
    aae(s6.intensities, [i ** 2 for i in range(1, 11)])


def test__intensities():
    energies, intensities = np.arange(1, 11), np.arange(1, 11)
    s1 = ConvSpectrum("Hello World", energies, intensities)

    aae(s1.intensities[1], s1._intensities(2))
    aae(s1.intensities[9], s1._intensities(10))
    aae(s1.intensities[2:4], s1._intensities(3, 5))

    with raises(IndexError):
        s1._intensities(-1)
    with raises(IndexError):
        s1._intensities(11)
    with raises(IndexError):
        s1._intensities(-1, 5)
    with raises(IndexError):
        s1._intensities(5, 11)
    with raises(IndexError):
        s1._intensities(11, 100)


def test_copy():
    energies, intensities = np.arange(1, 11), np.arange(1, 11)
    s1 = ConvSpectrum("Hello World", energies, intensities)
    s2 = s1.copy()
    assert s1 == s2
    assert id(s1) != id(s2)


def test_domain():
    energies, intensities = np.arange(10), np.arange(10)
    s1 = ConvSpectrum("Hello World", energies, intensities)

    assert s1.domain == (0, 9)


def test_smoothed():
    energies, intensities = np.arange(10), np.arange(1, 11)
    s1 = ConvSpectrum("Hello World", energies, intensities)
    s2 = s1.smoothed(3)

    aae(s1.energies, s2.energies)
    # Smoothing causes edge defects at the end
    aae(s1.intensities[:9], s2.intensities[:9])


def test_baseline_subtracted():
    energies, intensities = np.arange(10), np.arange(1, 11)
    s1 = ConvSpectrum("Hello World", energies, intensities)
    s2 = s1.baseline_subtracted()
    s3 = s1.baseline_subtracted(9)

    aae(s1.intensities - 1, s2.intensities)
    aae(s1.intensities - 9, s3.intensities)


def test_set_zero():
    energies, intensities = np.arange(10), np.arange(1, 11)
    s1 = ConvSpectrum("Hello World", energies, intensities)

    for val in [-1, 12]:
        with raises(IndexError):
            s1.set_zero(val)
    with raises(TypeError):
        s1.set_zero("a")
    for vals in [(-2, 2), (5, 13)]:
        with raises(IndexError):
            s1.set_zero(*vals)
    with raises(TypeError):
        s1.set_zero(5, "b")

    s2 = s1.set_zero(2)
    s3 = s1.set_zero(9)
    s4 = s1.set_zero(2, 5)

    aae(s1.intensities - 3, s2.intensities)
    aae(s1.intensities - 10, s3.intensities)
    aae(s1.intensities - 4, s4.intensities)
    aae(s1.intensities - 4, s4.intensities)


def test_sliced():
    energies, intensities = np.arange(10), np.arange(10)
    intensities2 = np.arange(5, 10)
    intensities3 = np.arange(5)
    s1 = ConvSpectrum("Hello World", energies, intensities)
    s2 = ConvSpectrum("Hello World", energies[5:], intensities2)
    s3 = ConvSpectrum("Hello World", energies[:5], intensities3)

    assert s1 == s1.sliced()
    assert s1 != s1.sliced(5)
    assert s2 == s1.sliced(5)
    assert s1 != s1.sliced(None, 5)
    assert s3 == s1.sliced(None, 5)


def test_from_csvs(tmp_path):
    test_csv = f"{tmp_path}/test.csv"
    with open(test_csv, "w") as f:
        f.write("x,A,B\n0,2,4\n1,3,5")
    ConvSpectrum.from_csvs(test_csv)
    ConvSpectrum.from_csvs("tests/files/xrd.csv")


def test_norm():
    energies, intensities = np.arange(10), np.arange(1, 11)
    s1 = ConvSpectrum("1", energies, intensities)
    aae(s1.norm, 19.6214168703)


def test_normed():
    energies, intensities = np.arange(10), np.arange(1, 11)
    s1 = ConvSpectrum("1", energies, intensities)

    aae(s1.normed("area").intensities, s1.intensities / 49.5)
    aae(s1.normed("max").intensities, s1.intensities / 10)
    aae(s1.normed(3).intensities, s1.intensities / 4)
    aae(s1.normed(4, 4).intensities, s1.intensities * 4 / 5)

    aae(s1.normed((3, 5)).intensities, s1.intensities / 10)
    with raises(IndexError):
        s1.normed((-1, 5))
    with raises(IndexError):
        s1.normed((5, 11))
    with raises(IndexError):
        s1.normed((-1, 50))
    with raises(ValueError):
        s1.normed((-1, "b"))


def test_peaks():
    spectrum = ConvSpectrum.from_csvs("tests/files/spectrum1.csv")[0]

    assert np.all(spectrum.peaks()[0] == [9, 13, 18, 21, 24])
    assert np.all(spectrum.peaks(indices=True)[0] == [4, 8, 13, 16, 19])


def test_min_max():
    spectrum = ConvSpectrum.from_csvs("tests/files/spectrum1.csv")[0]

    assert min(spectrum) == (5, 0)
    assert max(spectrum) == (25, 0)
    assert spectrum.min == (16, -10)
    assert spectrum.max == (13, 21)


def test_correlation():
    energies, intensities = np.arange(10), np.arange(1, 11)
    s1 = ConvSpectrum("1", energies, intensities)
    s2 = ConvSpectrum("2", np.array([0, 1, 2, 3]), np.array([1, 0, 1, 0]))
    s3 = ConvSpectrum("3", np.array([0, 1, 2, 3]), np.array([1, 1, 0, 0]))

    aae(s1.correlation(s1), 1)
    aae(s2.correlation(s3), 0.5)

    with raises(NotImplementedError):
        s1.correlation(s2)
