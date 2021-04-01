from pytest import raises

from spectra.fit import IR_guess_model, XRD_guess_model, fit_spectrum, guess_model, plot_fit
from spectra.spectrum import spectra_from_csvs


def setup():
    pass


def teardown():
    pass


def test_guess_model():
    spectrum = spectra_from_csvs("tests/files/spectrum1.csv")[0]
    spectrum.style = "XRD"

    model1, params1 = guess_model(spectrum)
    model2, params2 = XRD_guess_model(spectrum)
    model3, params3 = guess_model(spectrum, "IR")

    assert len(params1) == 38
    assert params1 == params2
    assert len(params3) == 15


def test_XRD_guess_model():
    spectrum = spectra_from_csvs("tests/files/xrd.csv")[0]

    model, params = XRD_guess_model(spectrum)
    assert len(params) == 38


def test_IR_guess_model():
    spectrum = spectra_from_csvs(
        "tests/files/1-butanol + N 3400/1.00% T12/Round 1/" + "Thu Jul 25 14-53-51 2019 (GMT-04-00).CSV"
    )[0]

    model, params = IR_guess_model(spectrum)
    assert len(params) == 65


def test_fit_spectrum(tmp_path):
    spectrum = spectra_from_csvs("tests/files/spectrum1.csv")[0]

    with raises(NotImplementedError):
        fit_spectrum(spectrum)

    spectrum.style = "GC/MS"
    with raises(NotImplementedError):
        fit_spectrum(spectrum)

    spectrum.style = "XRD"
    fit = fit_spectrum(spectrum)
    assert len(fit.params) == 38


def test_plot_fit():
    spectrum = spectra_from_csvs("tests/files/spectrum1.csv")[0]
    spectrum.style = "IR"
    fit_IR = fit_spectrum(spectrum)
    fit_XRD = fit_spectrum(spectrum, "XRD")

    fig, ax = plot_fit(fit_IR, spectrum.style)
    plot_fit(fit_XRD, "XRD", plot=(fig, ax), verbose=True, title="XRD - test_plot_fit")

    with raises(TypeError):
        fig, ax = plot_fit(fit_XRD, spectrum.style)
