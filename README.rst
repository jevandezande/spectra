=======
spectra
=======

.. image:: https://img.shields.io/github/license/jevandezande/spectra
    :target: https://github.com/jevandezande/spectra/blob/master/LICENSE
    :alt: MIT license

.. image:: https://img.shields.io/github/workflow/status/jevandezande/spectra/Test
    :target: https://github.com/jevandezande/spectra/actions
    :alt: GitHub Workflow Status

.. image:: https://img.shields.io/codecov/c/github/jevandezande/spectra
    :target: https://app.codecov.io/gh/jevandezande/spectra
    :alt: Codecov

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    :alt: black


Spectra is a package for analyzing and plotting various 1D spectra seen in
chemistry. Spectra can be algebraically manipulated and a variety of helper
functions are included to track peak progress, convolve spectra with various
shapes, and perform peak fitting.

It currently supports plotting the following spectra, but more can easily be added.

- GC
- HPLC
- MS
- NMR
- IR
- Raman
- UV-Vis
- XRD
- XPS


Samples
-------
.. image:: samples/IR/plots/ir_zsh.svg
    :width: 800px
    :align: center
    :alt: IR Plot

.. image:: samples/reaction_kinetics/plots/reaction_kinetics.svg
    :width: 800px
    :align: center
    :alt: Reaction Kinetics Plot

Contributing
------------
Additional contributions and suggestions are welcome. The intent is to produce
a small, easily usable package for producing quality graphics and assisting in
workflows. Speed is important, as `progress()` regularly needs to read and
manipulate 100+ Spectra at a time.
