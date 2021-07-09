import numpy as np
from numpy.testing import assert_almost_equal as aae
from pytest import mark, raises

from spectra import ConvSpectrum
from spectra.fit import IR_guess_model, XRD_guess_model, fit_spectrum, fit_with_spectra, guess_model, plot_fit


def setup():
    pass


def teardown():
    pass


def test_guess_model():
    spectrum = ConvSpectrum.from_csvs("tests/files/spectrum1.csv")[0]
    spectrum.style = "XRD"

    model1, params1 = guess_model(spectrum)
    model2, params2 = XRD_guess_model(spectrum)
    model3, params3 = guess_model(spectrum, "IR")

    assert len(params1) == 38
    assert params1 == params2
    assert len(params3) == 15


def test_XRD_guess_model():
    spectrum = ConvSpectrum.from_csvs("tests/files/xrd.csv")[0]

    model, params = XRD_guess_model(spectrum)
    assert len(params) == 38


def test_IR_guess_model():
    spectrum = ConvSpectrum.from_csvs(
        "tests/files/1-butanol + N 3400/1.00% T12/Round 1/Thu Jul 25 14-53-51 2019 (GMT-04-00).CSV"
    )[0]

    model, params = IR_guess_model(spectrum)
    assert len(params) == 65


def test_fit_spectrum():
    spectrum = ConvSpectrum.from_csvs("tests/files/spectrum1.csv")[0]

    with raises(NotImplementedError):
        fit_spectrum(spectrum)

    spectrum.style = "GC/MS"
    with raises(NotImplementedError):
        fit_spectrum(spectrum)

    spectrum.style = "XRD"
    fit = fit_spectrum(spectrum)
    assert len(fit.params) == 38


def test_plot_fit():
    spectrum = ConvSpectrum.from_csvs("tests/files/spectrum1.csv")[0]
    spectrum.style = "IR"
    fit_IR = fit_spectrum(spectrum)
    fit_XRD = fit_spectrum(spectrum, "XRD")

    fig, ax = plot_fit(fit_IR, spectrum.style)
    plot_fit(fit_XRD, "XRD", plot=(fig, ax), verbose=True, title="XRD - test_plot_fit")

    with raises(TypeError):
        fig, ax = plot_fit(fit_XRD, spectrum.style)


def test_fit_spectra():
    spectrum1 = ConvSpectrum.from_csvs("tests/files/spectrum1.csv")[0]
    spectrum2 = (spectrum1 + 5) / 2
    spectrum3 = spectrum1.copy()
    spectrum3.intensities = np.ones_like(spectrum3.intensities)

    spectrum4 = ConvSpectrum.from_csvs(
        "tests/files/1-butanol + N 3400/1.00% T12/Round 1/Thu Jul 25 14-53-51 2019 (GMT-04-00).CSV"
    )[0]
    spectrum5 = spectrum4.copy()
    spectrum5.intensities = np.ones_like(spectrum5.intensities)

    target1 = spectrum1.copy()
    target2 = spectrum1 + spectrum2
    target3 = spectrum2 + spectrum3 * 4
    target4 = spectrum1 - spectrum2

    target5 = spectrum4 + spectrum5 * 2

    aae([1, 0], fit_with_spectra(target1, spectrum1, spectrum2).x)

    aae([1, 1], fit_with_spectra(target2, spectrum1, spectrum2).x)
    aae([1, 1], fit_with_spectra(target2, spectrum1, spectrum2, x0=[0, 2]).x)
    aae([1, 1], fit_with_spectra(target2, spectrum1, spectrum2, x0=[3, -1]).x)

    aae([1, 4], fit_with_spectra(target3, spectrum2, spectrum3, x0=[-0.1, -1]).x)

    # Bounds default does not allow negatives
    aae(0, fit_with_spectra(target4, spectrum1, spectrum2, x0=[3, -2]).x[1])
    aae([1, -1], fit_with_spectra(target4, spectrum1, spectrum2, x0=[3, -2], bounds=None).x)

    aae([1, 2], fit_with_spectra(target5, spectrum4, spectrum5, x0=[5, -2]).x)


@mark.xfail
def test_multi_fit_spectra():
    spectrum1 = ConvSpectrum.from_csvs("tests/files/spectrum1.csv")[0]
    spectrum2 = (spectrum1 + 5) / 2
    spectrum3 = spectrum1.copy()
    spectrum3.intensities = np.ones_like(spectrum3.intensities)

    target = spectrum1 + spectrum2 + spectrum3

    options = {
        "maxiter": 10000,
        "ftol": 1e-11,
    }
    aae([1, 1, 1], fit_with_spectra(target, spectrum1, spectrum2, spectrum3, x0=[2.1, 0.5, 1.5], options=options).x)
