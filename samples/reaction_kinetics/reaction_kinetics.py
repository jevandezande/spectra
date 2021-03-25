import matplotlib.pyplot as plt

from spectra.reaction_kinetics import plot_reaction_kinetics

"""
Sample of reaction kinetics plitting for the 1-butanol + N 3400 (diisocyanate)
reaction catalyzed by T12.
"""

title = "Reaction Kinetics - 1-butanol + N 3400"
folder = "samples/reaction_kinetics/1-butanol + N 3400"
catalysts = ["0.01% T12", "0.03% T12", "0.10% T12", "0.30% T12", "1.00% T12"]
reactions = [f"{catalyst}" for catalyst in catalysts]

plot_reaction_kinetics(
    reactions,
    folder,
    names=catalysts,
    kinetics_norms=True,
    verbose=True,
    spectra_smooth=10,
    title=title,
    kinetics_xmax=60,
    savefig="reaction_kinetics.svg",
    combo_plot="only",
    spectra_plot=False,
)

plt.show()
