Computing Spectra
=================


Spectra are ultimately a combination of transitions happening at various energies and intensities.
Condensed phase effects can [broaden these transitions](https://en.wikipedia.org/wiki/Spectral_line#Line_broadening_and_shift), and thus the spectra computed in the gas phase are often convolved with a gaussian curve to approximate this effect.
Once the energies and intensities of a spectrum have been computed with the chosen program, plotting the results is as simple as:

```python
from spectra import SticksSpectrum
from spectra.plot import plotter, plot_spectrum

def plot(name, energies, intensities):
    spectrum = SticksSpectrum(name, energies, intensities)

    fig, ax = plotter(
        [spectrum],
        title=f"{name} IR Spectrum",
        style="IR",
        peaks=True,
        savefig=f"{name}_ir_spectrum.svg",
    )

    # Plot spectrum convolved with a Gaussian curve
    plot_spectrum(spectrum.convert(100), "IR", ax, label="Broadened")
```


![Example image of water IR spectrum](images/pyscf_water_ir_spectrum.svg)
