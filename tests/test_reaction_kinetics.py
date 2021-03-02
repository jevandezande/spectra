import sys

from pytest import raises

sys.path.insert(0, "..")

from spectra.reaction_kinetics import plot_reaction_kinetics


def setup():
    pass


def teardown():
    pass


def test_plot_reaction_kinetics(tmp_path):
    fig, axes = plot_reaction_kinetics([], "", verbose=True, title="Hello World")
    assert len(axes) == 1  # combo plot

    fig, axes = plot_reaction_kinetics(
        [], "", verbose=True, rounds=[3], savefig=f"{tmp_path}/a.svg"
    )

    with raises(ValueError):
        plot_reaction_kinetics([], "", colors=["red", "blue"])


def test_plot_reaction_kinetics_full(tmp_path):
    fig, axes = plot_reaction_kinetics([], "", verbose=True)
    assert len(axes) == 1  # combo plot
    file_dir = "tests/files"

    with raises(ValueError):
        plot_reaction_kinetics([], "", colors=["red", "blue"])

    catalysts = ["1.00% T12"]
    reactions = [f"1-butanol + N 3400/{catalyst}" for catalyst in catalysts]
    fig, axes = plot_reaction_kinetics(
        reactions, file_dir, verbose=True, spectra_norms=("max", 1), spectra_smooth=True
    )
    assert axes.shape == (2, 2)

    catalysts = ["1.00% T12", "0.10% T12"]  # '0.30% T12', '1.00% T12']
    reactions = [f"1-butanol + N 3400/{catalyst}" for catalyst in catalysts]
    fig, axes = plot_reaction_kinetics(
        reactions, file_dir, verbose=True, rounds=[5], savefig=f"{tmp_path}/a.svg"
    )
    assert axes.shape == (3, 2)
