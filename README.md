Spectra
=======

[![License](https://img.shields.io/github/license/jevandezande/spectra)](https://github.com/jevandezande/spectra/blob/master/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/jevandezande/spectra/test.yml?branch=master)](https://github.com/jevandezande/spectra/actions/)
[![Codecov](https://img.shields.io/codecov/c/github/jevandezande/{{cookiecutter.repo_name}})](https://codecov.io/gh/jevandezande/spectra)


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
![IR Plot](samples/IR/plots/ir_zsh.svg)

![Reaction Kinetics Plot](samples/reaction_kinetics/plots/reaction_kinetics.svg)

Contributing
------------
Additional contributions and suggestions are welcome. The intent is to produce
a small, easily usable package for producing quality graphics and assisting in
workflows. Speed is important, as `progress()` regularly needs to read and
manipulate 100+ Spectra at a time.
