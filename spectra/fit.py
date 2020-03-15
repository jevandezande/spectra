import numpy as np

from spectra.plot import setup_axis
from spectra.tools import integrate

import matplotlib.pyplot as plt

from lmfit import models


def fit_spectrum(spectrum, style=None, model=None, params=None, peak_args=None):
    """
    Fit a given spectrum.

    Note: guessing the fit is useful for determining the initial fit, but it is
    always recommended to perform multiple rounds of observing and
    hand-optimizing the fit to ensure it is performed properly.

    :param spectrum: the spectrum to be fit
    :param style: the style of spectrum (used as a hint for guessing the fit)
    :param model: model to be used
    :param params: params used in the model
    :param peak_args: peak picking arguments to be used for guessing the model
    :return: model fit
    """
    if model is None:
        model, params = guess_model(spectrum, style, peak_args)

    return model.fit(spectrum.ys, params, x=spectrum.xs)


def guess_model(spectrum, style=None, peak_args=None):
    """
    Return a guess model of the correct style.

    :param spectrum: the spectrum to be fit
    :param style: the style of spectrum (used as a hint for guessing the fit)
    :param peak_args: peak picking arguments to be used for guessing the model
    :return: model, params
    """
    style = spectrum.style if style is None else style
    peak_args = {} if peak_args is None else peak_args

    if style == 'XRD':
        return XRD_guess_model(spectrum, peak_args)
    else:
        raise NotImplementedError(f'Does not yet know how to guess a fit for {style}.')


def XRD_guess_model(spectrum, peak_args=None):
    """
    Guess a fit for the XRD based on the peaks in the spectrum.

    :param spectrum: the spectrum to be fit
    :param peak_args: arguments for finding peaks
    :return: a model to feed to optimizer
    """
    min_x = spectrum.xs[0]
    max_x = spectrum.xs[-1]
    range_x = max_x - min_x
    max_y = np.max(spectrum.ys)
    min_y = np.min(spectrum.ys)
    range_y = max_y - min_y

    XRD_peak_defaults = {
        'prominence': 0.02*range_y,
    }
    peak_args = XRD_peak_defaults if peak_args is None else {**XRD_peak_defaults, **peak_args}

    peak_indices, peak_properties = spectrum.peaks(**peak_args, indices=True)

    params = None
    composite_model = None

    # Fit the peaks
    i = 0
    for i, peak_idx in enumerate(peak_indices):
        prefix = f'c{i}_'
        model = models.VoigtModel(prefix=prefix)
        center = spectrum.xs[peak_idx]
        height = spectrum.ys[peak_idx]

        model.set_param_hint('amplitude', min=100, max=1.1*height)
        model.set_param_hint('center', min=center - 1, max=center + 1)
        model.set_param_hint('sigma', min=0.01, max=0.3)
        peak_params = {
            f'{prefix}amplitude': height*0.8,
            f'{prefix}center': center,
            f'{prefix}sigma': 0.1,
        }

        model_params = model.make_params(**peak_params)
        params = model_params if params is None else params.update(model_params)
        composite_model = model if composite_model is None else composite_model + model

    # Add a broad amorphous peak
    prefix = f'a{i+1}_'
    model = models.GaussianModel(prefix=prefix)
    model.set_param_hint('amplitude', min=1, max=max_y)
    model.set_param_hint('center', min=min_x, max=max_x)
    model.set_param_hint('sigma', min=0.1, max=range_x/2)
    peak_params = {
        f'{prefix}amplitude': max_y/4,
        f'{prefix}center': 20,
        f'{prefix}sigma': 5,
    }
    model_params = model.make_params(**peak_params)
    params = params.update(model_params)
    composite_model += model

    # Add a broader amorphous background peak
    prefix = f'b{i+2}_'
    model = models.ExponentialModel(prefix=prefix)
    model.set_param_hint('amplitude', min=1, max=max_y)
    #model.set_param_hint('decay', min=min_x, max=max_x)
    peak_params = {
        f'{prefix}amplitude': spectrum.ys[:10].mean()*0.8,
        f'{prefix}decay': 30,
    }
    model.set_param_hint('amplitude', min=1, max=spectrum.ys[:10].mean()*2)
    model.set_param_hint('decay', min=10, max=100)

    model_params = model.make_params(**peak_params)
    params = model_params if params is None else params.update(model_params)
    composite_model = model if composite_model is None else composite_model + model

    return composite_model, params


def plot_fit(fit, style, plot=None, verbose=False, **setup_axis_args):
    """
    Plot the results of fitting a spectrum.

    :param fit: the results of lmfit.model.fit()
    :param style: the style of spectrum (used as a hint for guessing the fit)
    :param plot: (figure, axis) on which to plot, generates new figure if None
    :param verbose: print the parameters of the fit
    :param setup_axis_args: arguments to be passed to setup_axis
    """
    if plot is None:
        fig, ax = plt.subplots()
        setup_axis(ax, style, **setup_axis_args)
    else:
        fig, ax = plot

    xs = fit.userkws['x']
    ys = fit.data

    ax.scatter(xs, ys, s=1, label='Spectrum')
    ax.plot(xs, fit.best_fit, label='Optimized')

    components = fit.eval_components()
    if verbose:
        print(f"{'Function':12s}: Initial → Final: Portion of Total, Portion of Fit")
        for name in fit.init_values:
            print(f'{name:12s}: {fit.init_values[name]:7.2f} → {fit.best_values[name]:7.2f}')

    if style == 'XRD':
        area = {
            'crystalline': 0,
            'amorphous': 0,
            'background': 0,
            'initial': integrate(xs, fit.init_fit),
            'optimized': integrate(xs, fit.best_fit),
            'total': integrate(xs, ys),
        }
    else:
        raise NotImplementedError(f'Does not yet know how to plot a fit for {style}.')

    if verbose:
        print()
        print("Name        | Ratio of fit area | Ratio of total area (-background)")
        print('-'*67)

    for name, vals in components.items():
        name = name[:-1]
        peak_area = integrate(xs, vals)
        if name[0] == 'c':
            area['crystalline'] += peak_area
            linestyle = '-'
        elif name[0] == 'a':
            area['amorphous'] += peak_area
            linestyle = '--'
        elif name[0] == 'b':
            area['background'] += peak_area
            linestyle = '--'
        else:
            raise ValueError(f'Not sure what to do with peak named: {name}')
        ax.plot(xs, vals, linestyle=linestyle, label=name)

        if verbose:
            print(f"{name:11s} |      {peak_area/area['optimized']:>6.3f}       |      {peak_area/(area['total'] - area['background']):>6.3f}")

    if verbose:
        print('-'*67)
        for name, n_area in area.items():
            print(f"{name:11s} |      {n_area/area['optimized']:>6.3f}       |      {n_area/(area['total'] - area['background']):>6.3f}")
        print()

    fig.legend()

    return fig, ax
