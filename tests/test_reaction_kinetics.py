import sys


from pytest import raises

sys.path.insert(0, '..')

from spectra.reaction_kinetics import plot_reaction_kinetics


def setup():
    pass


def teardown():
    pass


def test_plot_reaction_kinetics():
    fig, axes = plot_reaction_kinetics([], '', verbose=True)
    assert len(axes) == 1  # combo plot

    fig, axes = plot_reaction_kinetics([], '', verbose=True, rounds=[3])

    with raises(ValueError):
        plot_reaction_kinetics([], '', colors=['red', 'blue'])


def test_plot_reaction_kinetics_full():
    fig, axes = plot_reaction_kinetics([], '', verbose=True)
    assert len(axes) == 1  # combo plot

    with raises(ValueError):
        plot_reaction_kinetics([], '', colors=['red', 'blue'])

    catalysts = ['0.01% T12']
    reactions = [f'1-butanol + N 3400/{catalyst}' for catalyst in catalysts]
    fig, axes = plot_reaction_kinetics(reactions, 'tests/files/', verbose=True)
    assert axes.shape == (2, 2)

    catalysts = ['0.01% T12', '0.03% T12']#, '0.10% T12', '0.30% T12', '1.00% T12']
    reactions = [f'1-butanol + N 3400/{catalyst}' for catalyst in catalysts]
    fig, axes = plot_reaction_kinetics(reactions, 'tests/files/', verbose=True, rounds=[5])
    assert axes.shape == (3, 2)
