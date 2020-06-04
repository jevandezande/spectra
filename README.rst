=======
spectra
=======


.. image:: https://img.shields.io/travis/jevandezande/spectra.svg
        :target: https://travis-ci.com/jevandezande/spectra

.. image:: https://readthedocs.org/projects/spectra/badge/?version=latest
        :target: https://spectra.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status



Spectra is a package for analyzing and plotting various 1D spectra seen in
chemistry. It currently supports GC, HPLC, MS, NMR, FTIR, Raman, UV-Vis, XRD,
and XPS. Spectra can be be subjected to various algebraic manipulations and
integrated across slices, allowing for the plotting of a peaks progress over
time using `progress()`. Curve fitting can be performed, with intelligent
guesses implemented for a few different spectra types. Various utilities for
plotting Spectra are available, along with a large range of ways to adjust the
output figures.

Currently, only data in CSVs is supported.


* Free software: MIT license
* Documentation: https://spectra.readthedocs.io.


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
