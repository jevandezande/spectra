import numpy as np
from numpy.testing import assert_almost_equal as aae
from pytest import mark, raises

from spectra import ConvSpectrum, SticksSpectrum
from spectra.tools import (
    boltzmann_factors,
    boltzmann_weighted,
    cull,
    glob_read_csvs,
    index_of_x,
    integrate,
    read_csv,
    read_csvs,
    y_at_x,
)


def setup():
    pass


def teardown():
    pass


def test_read_csv(tmp_path):
    path = f"{tmp_path}/test.csv"
    data = [["A", "B", "C", "D"], [1, 2, 3, 4], [5, 6, 7, 8]]

    with open(path, "w") as f:
        f.write("\n".join(",".join(map(str, row)) for row in data))

    csv = read_csv(path)
    assert csv[0] == data[0]
    aae(csv[1], [1, 5])
    aae(csv[2], [[2, 6], [3, 7], [4, 8]])

    csv = read_csv("tests/files/1-butanol + N 3400/1.00% T12/Round 1/Thu Jul 25 14-53-51 2019 (GMT-04-00).CSV")


def test_read_csvs(tmpdir):
    p1 = tmpdir.mkdir("sub").join("test1.csv")
    p2 = tmpdir.join("test2.csv")
    data1 = [["A", "B", "C", "D"], [1, 2, 3, 4], [5, 6, 7, 8]]
    data2 = [["A", "B"], [6, 7], [8, 9]]

    data_str1 = "\n".join(",".join(map(str, row)) for row in data1)
    data_str2 = "\n".join(",".join(map(str, row)) for row in data2)
    p1.write(data_str1)
    p2.write(data_str2)
    assert len(tmpdir.listdir()) == 2

    read_csvs(str(p1))
    read_csvs([str(p1)])
    read_csvs(str(p2))
    titles, energies, intensities = read_csvs([p1, p2])
    assert titles == ["B", "C", "D", "B"]
    aae(energies, [[1, 5], [1, 5], [1, 5], [6, 8]])
    aae(intensities, [[2, 6], [3, 7], [4, 8], [7, 9]])


def test_glob_read_csvs():
    file_dir1 = "tests/files/1-butanol + N 3400/1.00% T12/Round 1"
    titles, energies, intensities, file_names = glob_read_csvs(f"{file_dir1}/Thu Jul 25 14*.CSV")
    assert len(file_names) == 18

    file_dir2 = "tests/files/1-butanol + N 3400/0.03% T12/Round 1"
    titles, energies, intensities, file_names = glob_read_csvs([f"{file_dir2}/Thu Aug 01 08-49*.CSV"])
    assert len(file_names) == 3


def test_index_of_x():
    with raises(IndexError):
        index_of_x(0, [])
    with raises(IndexError):
        index_of_x(0, [1, 2, 3])

    assert 0 == index_of_x(1, [1, 2, 3])
    assert 2 == index_of_x(1, [3, 2, 1])


def test_y_at_x():
    energies = [1, 2, 3]
    intensities = [4, 5, 6]
    with raises(IndexError):
        y_at_x(0, [], [])

    with raises(ValueError):
        y_at_x(0, energies, [])

    with raises(IndexError):
        y_at_x(0, energies, intensities)

    assert 4 == y_at_x(1, energies, intensities)
    assert 5 == y_at_x(2, energies, intensities)
    assert 6 == y_at_x(2.5, energies, intensities)
    assert 4 == y_at_x(3, energies[::-1], intensities)
    assert 5 == y_at_x(2, energies[::-1], intensities)
    assert 4 == y_at_x(2.5, energies[::-1], intensities)

    assert y_at_x(2.1, energies, intensities) == y_at_x(2.1, energies[::-1], intensities[::-1])


def test_integrate():
    assert 0 == integrate([], [])

    with raises(ValueError):
        integrate([1, 2, 3], [])

    assert 10 == integrate([1, 2, 3], [4, 5, 6])

    with raises(IndexError):
        integrate([1, 2, 3], [4, 5, 6], [0, 1])
    with raises(IndexError):
        integrate([1, 2, 3], [4, 5, 6], [1, 9])

    assert 4.5 == integrate([1, 2, 3], [4, 5, 6], [1, 1.1])


def test_cull():
    assert list(cull(range(10), 3)) == [0, 5, 9]


def test_boltzmann_factors():
    aae(boltzmann_factors(np.zeros(10)), [0.1] * 10)
    aae(boltzmann_factors(np.ones(10)), [0.1] * 10)

    with raises(ZeroDivisionError):
        boltzmann_factors(np.arange(10), 0)

    aae(boltzmann_factors(np.arange(10), 1e-10), [1] + [0] * 9)
    aae(boltzmann_factors(np.arange(10), 1e99), [0.1] * 10)

    aae(
        boltzmann_factors(-100 + np.linspace(0, 0.001, 11)),
        [
            0.14567326,
            0.13111933,
            0.11801945,
            0.10622836,
            0.09561529,
            0.08606256,
            0.07746422,
            0.06972492,
            0.06275884,
            0.05648873,
            0.05084505,
        ],
    )


def setup_spectra(cls):
    s1 = cls("S1", np.arange(10), np.arange(10))
    s2 = cls("S2", np.arange(10), -np.arange(10))
    s3 = cls("S3", np.arange(10), np.ones(10))
    return [s1, s2, s3]


@mark.parametrize("Spec", [ConvSpectrum, SticksSpectrum])
def test_boltzmann_basic(Spec):
    spectra = setup_spectra(Spec)

    with raises(AssertionError):
        assert boltzmann_weighted([], [])

    with raises(AssertionError):
        assert boltzmann_weighted(spectra, [2, 3])

    with raises(ZeroDivisionError):
        boltzmann_weighted(spectra, np.arange(3), 0)

    boltzmann_weighted(spectra, np.zeros(3)) == boltzmann_weighted(spectra, np.ones(3))


def test_boltzmann_weighted_conv_spectrum():
    spectra = setup_spectra(ConvSpectrum)
    energies = [-1.002, -1.001, -1.000]

    S = boltzmann_weighted([spectra[0]] * 3, np.zeros(3), T=300)
    aae(S.energies, spectra[0].energies)
    aae(S.intensities, spectra[0].intensities)

    # At low temperatures, only the lowest energy result matters
    aae(boltzmann_weighted(spectra, energies, 1).intensities, range(10))

    # At high temperature, energy differences become insignificant
    aae(boltzmann_weighted(spectra, energies, 1e10).intensities, [1 / 3] * 10)


def test_boltzmann_weighted_sticks_spectrum():
    spectra = setup_spectra(SticksSpectrum)
    energies = [-1.002, -1.001, -1.000]

    S = boltzmann_weighted([spectra[0]] * 3, np.zeros(3), T=300)
    aae(S.energies, list(spectra[0].energies) * 3)
    aae(S.intensities, list(spectra[0].intensities / 3) * 3)

    # At low temperatures, only the lowest energy result matters
    aae(
        boltzmann_weighted(spectra, energies, 1).intensities,
        list(range(10)) + [0] * 20,
    )

    # At high temperature, energy differences become insignificant
    aae(
        boltzmann_weighted(spectra, energies, 1e10).intensities,
        list(np.arange(10) / 3) + list(-np.arange(10) / 3) + list(np.ones(10) / 3),
    )
