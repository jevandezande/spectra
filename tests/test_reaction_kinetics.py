import sys

sys.path.insert(0, '..')

from spectra.reaction_kinetics import plot_reaction_kinetics
from pytest import raises


def setup():
    pass


def teardown():
    pass


def test_plot_reaction_kinetics():
    fig, axes = plot_reaction_kinetics([], [], '')
    assert len(axes) == 1  # combo plot

    with raises(ValueError):
        plot_reaction_kinetics([], [], '', colors=['red', 'blue'])
