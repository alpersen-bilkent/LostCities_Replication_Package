# -*- coding: utf-8 -*-
"""
Figure (Appendix A, fig:knowncities): overview map of the 15 known Bronze
Age cities on the Anatolian basemap. Purely descriptive of the input data
-- draws city names/coordinates directly from src/data.py's name-keyed
lookup (data.ancient_known_coords(), backed by _ancient_coords_lookup()),
the same source every other script in this package uses, so it cannot
drift out of sync the way an old, unrecovered ad-hoc script apparently did
(see data.ancient_known_lost_split()'s docstring for the historical bug
this class of mistake caused elsewhere in the project).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt  # noqa: E402

from src import config, data, plots  # noqa: E402


def main():
    known_coords, known_names = data.ancient_known_coords()


    fig, ax = plt.subplots(figsize=plots.FIGSIZE)
    plots.load_and_plot_turkey_map(ax)
    ax.set_xlim(31, 39.2)
    ax.set_ylim(35.8, 42.25)
    ax.set_aspect('equal')

    lons = known_coords[:, 0]
    lats = known_coords[:, 1]
    ax.scatter(lons, lats, c='blue', s=80, edgecolors='k', zorder=5)

    # Hanaknak (35.817, 40.0) and Karahna (36.1, 40.0) sit only ~0.28 deg
    # apart at the same latitude, so their default centered-above labels
    # collide. Push each one sideways, away from the other, instead of
    # stacking them both centered on the same spot; every other city is
    # spaced out enough for the default centered-above placement.
    label_overrides = {
        'Hanaknak': dict(dx=-0.08, dy=0.01, ha='right'),
        'Karahna': dict(dx=0.05, dy=0.01, ha='left'),
    }
    for name, lon, lat in zip(known_names, lons, lats):
        override = label_overrides.get(name, dict(dx=0.0, dy=0.07, ha='center'))
        ax.text(lon + override['dx'], lat + override['dy'], name, fontsize=8,
                 color='blue', fontweight='bold', ha=override['ha'], zorder=5)

    ax.set_title("Geographic Distribution of 15 Known Cities", fontsize=16)

    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(config.FIGURES_DIR, "KnownCitiesTurkey.png")
    plt.savefig(out_path, dpi=plots.DPI, bbox_inches='tight', pad_inches=0.1)
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
