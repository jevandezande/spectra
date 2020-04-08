#!/usr/bin/env zsh

PYTHONPATH=../../:$PYTHONPATH
PATH=../../bin:$PATH

plot_spectra -i data/{HDI,methanol}.csv -s plots/ir_zsh.svg -n 'HDI' 'Methanol' --title 'IR Spectra' -p
