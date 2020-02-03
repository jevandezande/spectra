import sys

from numpy.testing import assert_almost_equal as aae

from pytest import raises

sys.path.insert(0, '..')

from spectra.spectrum import *


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

    with raises(NotImplementedError):
        s = s1.copy()
        s.xs += 1
        s + s1

    aae(s1.xs, s2.xs)
    aae(s1.xs, s3.xs)
    aae(s1.xs, s4.xs)
    aae(s2.ys, np.arange(1, 11))
    aae(s3.ys, s1.ys)
    aae(s4.ys, 1 - s1.ys)


def test_abs():
    xs, ys, ys2 = np.arange(10), np.arange(10), np.arange(10)
    ys2[5:] = -ys2[5:]
    s1 = Spectrum('Hello World', xs, ys)
    s2 = Spectrum('Hello World', xs, ys2)

    assert s1 != s2
    assert s1 == abs(s2)


def test_mul():
    xs, ys = np.arange(10), np.arange(10)
    s1 = Spectrum('Hello World', xs, ys)
    s2 = 2 * s1
    s3 = s2 * 0.5

    with raises(NotImplementedError):
        s = s1.copy()
        s.xs += 1
        s * s1

    aae(s1.xs, s2.xs)
    aae(s1.xs, s3.xs)
    aae(s2.ys, np.arange(0, 20, 2))
    aae(s3.ys, s1.ys)


def test_div():
    xs, ys = np.arange(1, 11), np.arange(1, 11)
    s1 = Spectrum('Hello World', xs, ys)
    s2 = 1 / s1
    s3 = s1 / 2
    s4 = s1 / s1
    s5 = Spectrum('Hello World', xs, np.array([1]*10))
    s6 = s1 / s2

    aae(s1.xs, s2.xs)
    aae(s1.xs, s3.xs)
    aae(s2.ys, 1 / s1.ys)
    aae(s3.ys, np.arange(1, 11) / 2)
    aae(s4.ys, s5.ys)
    aae(s6.ys, [i**2 for i in range(1, 11)])


def test__ys():
    xs, ys = np.arange(1, 11), np.arange(1, 11)
    s1 = Spectrum('Hello World', xs, ys)

    aae(s1.ys[1], s1._ys(2))
    aae(s1.ys[9], s1._ys(10))
    aae(s1.ys[2:4], s1._ys(3, 5))

    with raises(IndexError):
        s1._ys(-1)
    with raises(IndexError):
        s1._ys(11)
    with raises(IndexError):
        s1._ys(-1, 5)
    with raises(IndexError):
        s1._ys(5, 11)
    with raises(IndexError):
        s1._ys(11, 100)


def test_copy():
    xs, ys = np.arange(1, 11), np.arange(1, 11)
    s1 = Spectrum('Hello World', xs, ys)
    s2 = s1.copy()
    assert s1 == s2
    assert id(s1) != id(s2)


def test_domain():
    xs, ys = np.arange(10), np.arange(10)
    s1 = Spectrum('Hello World', xs, ys)

    assert s1.domain == (0, 9)


def test_smoothed():
    xs, ys = np.arange(10), np.arange(1, 11)
    s1 = Spectrum('Hello World', xs, ys)
    s2 = s1.smoothed(3)

    aae(s1.xs, s2.xs)
    # Smoothing causes edge defects at the end
    aae(s1.ys[:9], s2.ys[:9])


def test_baseline_subtracted():
    xs, ys = np.arange(10), np.arange(1, 11)
    s1 = Spectrum('Hello World', xs, ys)
    s2 = s1.baseline_subtracted()
    s3 = s1.baseline_subtracted(9)

    aae(s1.ys - 1, s2.ys)
    aae(s1.ys - 9, s3.ys)


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

    aae(s1.ys - 3, s2.ys)
    aae(s1.ys - 10, s3.ys)
    aae(s1.ys - 4, s4.ys)
    aae(s1.ys - 4, s4.ys)


def test_sliced():
    xs, ys = np.arange(10), np.arange(10)
    ys2 = np.arange(5, 10)
    ys3 = np.arange(5)
    s1 = Spectrum('Hello World', xs, ys)
    s2 = Spectrum('Hello World', xs[5:], ys2)
    s3 = Spectrum('Hello World', xs[:5], ys3)

    assert s1 == s1.sliced()
    assert s1 != s1.sliced(5)
    assert s2 == s1.sliced(5)
    assert s1 != s1.sliced(None, 5)
    assert s3 == s1.sliced(None, 5)


def test_from_csvs(tmp_path):
    test_csv = f'{tmp_path}/test.csv'
    with open(test_csv, 'w') as f:
        f.write('x,A,B\n0,2,4\n1,3,5')
    spectra_from_csvs(test_csv)


def test_normed():
    xs, ys = np.arange(10), np.arange(1, 11)
    s1 = Spectrum('Hello World', xs, ys)

    aae(s1.normed('max').ys, s1.ys/10)
    aae(s1.normed(3).ys, s1.ys/4)
    aae(s1.normed(4, 4).ys, s1.ys*4/5)

    aae(s1.normed((3, 5)).ys, s1.ys/10)
    with raises(IndexError):
        s1.normed((-1, 5))
    with raises(IndexError):
        s1.normed((5, 11))
    with raises(IndexError):
        s1.normed((-1, 50))
    with raises(ValueError):
        s1.normed((-1, 'b'))


def test_peaks():
    spectrum = spectra_from_csvs('tests/files/spectrum1.csv')[0]

    assert np.all(spectrum.peaks()[0] == [9, 13, 18, 21, 24])
    assert np.all(spectrum.peaks(indices=True)[0] == [4, 8, 13, 16, 19])


def test_min_max():
    spectrum = spectra_from_csvs('tests/files/spectrum1.csv')[0]

    assert spectrum.min == (16, -10)
    assert spectrum.max == (13, 21)
