from importlib import import_module

import numpy as np

from spectra import SticksSpectrum
from spectra.plot import plot_spectrum, plotter


def main(
    filename: str = "water.zmat",
    program: str = "pyscf",
    functional: str = "BP86",
    basis_set: str = "def2-SVP",
    nstates: int = 10,
):
    name = filename.split(".")[0]
    with open(f"molecules/{filename}") as f:
        geom = f.read()

    program = program.lower()

    mod = import_module(f"{program}_runner")

    energies, intensities = mod.run_tddft(geom, functional, basis_set, nstates)

    plot(name, energies, intensities)


def plot(name: str, energies: np.ndarray, intensities: np.ndarray):
    spectrum = SticksSpectrum(name, energies, intensities)

    fig, ax = plotter(
        [spectrum],
        title=f"{name} UV-Vis Spectrum",
        style="UV-Vis",
        peaks=True,
        savefig=f"{name}_uv-vis_spectrum.svg",
    )

    # Plot spectrum convolved with a Gaussian curve
    plot_spectrum(spectrum.convert(10), "UV-Vis", ax, label="Broadened")

    return fig, ax


if __name__ == "__main__":
    main()
