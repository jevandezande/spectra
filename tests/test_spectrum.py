import sys

from numpy.testing import assert_almost_equal as aas
from pytest import raises

sys.path.insert(0, '..')

from spectra.spectrum import *
from spectra.tools import read_csv


def setup():
    pass


def teardown():
    pass


def test_init():
    xs, ys = np.arange(10), np.arange(10)
    Spectrum('Hello World', xs, ys)


def test_iter():
    xs, ys = np.arange(10), np.arange(10)
    s1 = Spectrum('Hello World', xs, ys)
    assert all(x == y for x, y in s1)


def test_eq():
    xs, ys = np.arange(10), np.arange(10)
    s1 = Spectrum('S1', xs, ys)
    s2 = Spectrum('S1', xs, ys)
    s3 = Spectrum('S3', xs, ys)
    s4 = Spectrum('S4', xs, xs)
    s5 = Spectrum('S5', ys, ys)

    assert s1 == s2
    assert s1 != s3
    assert s1 != s4
    assert s1 != s5


def test_len():
    xs, ys = np.arange(10), np.arange(10)
    s1 = Spectrum('Hello World', xs, ys)
    s2 = Spectrum('Hello World', ys, xs)
    assert len(s1) == len(xs)
    assert len(s2) == len(xs)


def test_str():
    xs, ys = np.arange(10), np.arange(10)
    s1 = Spectrum('Hello World', xs, ys)
    assert str(s1) == "<Spectrum: Hello World>"


def test_add_sub():
    xs, ys = np.arange(10), np.arange(10)
    s1 = Spectrum('Hello World', xs, ys)
    s2 = 1 + s1
    s3 = s2 - 1
    s4 = 1 - s3
    aas(s1.xs, s2.xs)
    aas(s1.xs, s3.xs)
    aas(s1.xs, s4.xs)
    aas(s2.ys, np.arange(1, 11))
    aas(s3.ys, s1.ys)
    aas(s4.ys, 1 - s1.ys)


def test_mul():
    xs, ys = np.arange(10), np.arange(10)
    s1 = Spectrum('Hello World', xs, ys)
    s2 = 2 * s1
    s3 = s2 * 0.5
    aas(s1.xs, s2.xs)
    aas(s1.xs, s3.xs)
    aas(s2.ys, np.arange(0, 20, 2))
    aas(s3.ys, s1.ys)


def test_div():
    xs, ys = np.arange(1, 11), np.arange(1, 11)
    s1 = Spectrum('Hello World', xs, ys)
    s2 = 1 / s1
    s3 = s1 / 2
    aas(s1.xs, s2.xs)
    aas(s1.xs, s3.xs)
    aas(s2.ys, 1 / s1.ys)
    aas(s3.ys, np.arange(1, 11) / 2)


def test_domain():
    xs, ys = np.arange(10), np.arange(10)
    s1 = Spectrum('Hello World', xs, ys)
    assert s1.domain == (0, 9)


def test_smoothed():
    xs, ys = np.arange(10), np.arange(1, 11)
    s1 = Spectrum('Hello World', xs, ys)
    s2 = s1.smoothed(3)
    aas(s1.xs, s2.xs)
    # Smoothing cause edge defects at the end
    aas(s1.ys[:9], s2.ys[:9])


def test_baseline_subtracted():
    xs, ys = np.arange(10), np.arange(1, 11)
    s1 = Spectrum('Hello World', xs, ys)
    s2 = s1.baseline_subtracted()
    s3 = s1.baseline_subtracted(9)
    aas(s1.ys - 1, s2.ys)
    aas(s1.ys - 9, s3.ys)


def test_set_zero():
    xs, ys = np.arange(10), np.arange(1, 11)
    s1 = Spectrum('Hello World', xs, ys)
    for val in [-1, 12]:
        with raises(IndexError):
            s1.set_zero(val)
    with raises(TypeError):
        s1.set_zero('a')
    for vals in [(-2, 2), (5, 13)]:
        with raises(IndexError):
            s1.set_zero(*vals)
    with raises(TypeError):
        s1.set_zero(5, 'b')

    s2 = s1.set_zero(2)
    s3 = s1.set_zero(9)
    s4 = s1.set_zero(2, 5)
    aas(s1.ys - 3, s2.ys)
    aas(s1.ys - 10, s3.ys)
    aas(s1.ys - 4, s4.ys)
    aas(s1.ys - 4, s4.ys)


def test_from_csvs(tmp_path):
    test_csv = f'{tmp_path}/test.csv'
    with open(test_csv, 'w') as f:
        f.write('x,A,B\n0,2,4\n1,3,5')
    spectra_from_csvs(test_csv)


def test_peaks():
    spectrum = spectra_from_csvs('files/spectrum1.csv')[0]
    assert np.all(spectrum.peaks()[0] == [9, 13, 18, 21, 24])
    assert np.all(spectrum.peaks(indices=True)[0] == [4, 8, 13, 16, 19])


def test_max_absorbance():
    spectrum = spectra_from_csvs('files/spectrum1.csv')[0]
    assert spectrum.max_absorbance == (13, 21)
