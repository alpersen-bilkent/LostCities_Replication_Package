# -*- coding: utf-8 -*-
"""
Prediction for the 10 actually-lost ancient cities, Dyadic-share
weights, full 25-node graph, supervised by all 15 known cities every
run, WITHOUT the directional-constraint penalty
(ancient_lost_dyadic_constrained.py has that).

This unconstrained run is now the robustness/ablation check, not the
paper's headline lost-city table: it shows what trade topology alone
predicts, with no outside textual evidence injected, complementing the
same claim already made by the LOO and modern-city results. The
constrained run in ancient_lost_dyadic_constrained.py is the reported
result (paper Table 11/12), since for the actual lost-city predictions
there is no reason to withhold legitimate historical-text evidence, and
that run satisfies every constraint and agrees more closely with all
three independent comparison estimates.

Corresponds to legacy/GCNDyadicBootstrapResults.py. LR=0.0053,
EMBEDDING_DIM=16 unchanged by the bug fixes. EPOCHS updated 1473 -> 250
per experiments/grid_search_ancient_dyadic.py, re-run on the corrected
25-node graph. N_BOOTSTRAP=200.

Bug 1 fix: this is the script whose weight formula and resampling base
diverged from the training script (legacy/GCNDyadicBootstrap.py). Both
now go through src/weights.py:dyadic_weights() on the same symmetrized
matrix -- see src/evaluate.py:run_ancient_lost_prediction().

Reports distance to three independent estimates, all distinct:
  - Barjamovic et al.'s structural gravity model (their QJE paper's own
    NLLS output) -- Dist_to_Baj, the original comparison.
  - Barjamovic's own 2011 historical-geography monograph proposal
    (philological judgment, not a fitted model) -- Dist_to_Barjamovic2011.
  - Forlanini's 2008 historical-geography proposal -- Dist_to_Forlanini2008.
The latter two are shown as 'B'/'F' letter markers on the individual
per-city plots only (plot_aggregate_map deliberately omits them --
10 cities x 3 extra markers each would be unreadable there).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from src import config, data, evaluate, plots, terrain  # noqa: E402

EXPERIMENT_NAME = "ancient_lost_dyadic"
MODEL_LABEL = "Dyadic"
CATEGORY = "Lost"
HPARAMS = {"epochs": 250, "lr": 0.0053, "embedding_dim": 16}
N_BOOTSTRAP = 200
ELLIPSE_N_STD = 2.0  # matches the 2-sigma ellipse drawn on the plots


def main():
    summary_df, raw_results = evaluate.run_ancient_lost_prediction(HPARAMS, N_BOOTSTRAP)

    out_path = evaluate.save_results(summary_df, EXPERIMENT_NAME)
    print(f"\nSaved: {out_path}")
    print("\n--- PREDICTION RESULTS ---")
    print("(Dist_to_Baj = Barjamovic et al. gravity model; "
          "Dist_to_Barjamovic2011 = Barjamovic's own historian proposal; "
          "Dist_to_Forlanini2008 = Forlanini's historian proposal)")
    print(summary_df.round(3).to_string(index=False))
    print(f"\nAvg Distance to Barjamovic et al. (gravity model): {summary_df['Dist_to_Baj'].mean():.2f} km")
    print(f"Avg Distance to Barjamovic 2011 (historian): {summary_df['Dist_to_Barjamovic2011'].mean():.2f} km")
    print(f"Avg Distance to Forlanini 2008 (historian): {summary_df['Dist_to_Forlanini2008'].mean():.2f} km")

    _, _, _, lost_names = data.ancient_known_lost_split()
    baj = data.barjamovic_lost_lookup()
    baj_2011 = data.barjamovic_2011_lookup()
    forlanini = data.forlanini_2008_lookup()
    known_coords, known_names = data.ancient_known_coords()
    known_ref = {name: tuple(known_coords[i]) for i, name in enumerate(known_names)}

    fig_dir = os.path.join(config.FIGURES_DIR, EXPERIMENT_NAME)
    known_sites_dir = os.path.join(config.RESULTS_DIR, "known_sites")
    minerals_dir = os.path.join(config.RESULTS_DIR, "minerals")
    os.makedirs(known_sites_dir, exist_ok=True)
    os.makedirs(minerals_dir, exist_ok=True)
    for name in lost_names:
        pts = np.array(raw_results[name])
        polygon, _ = terrain.geo_screened_region(pts, n_std=ELLIPSE_N_STD)
        known_sites = terrain.sites_within_polygon(polygon)
        known_sites.to_csv(os.path.join(known_sites_dir, f"{EXPERIMENT_NAME}_{name}.csv"), index=False)
        minerals = terrain.minerals_within_polygon(polygon)
        minerals.to_csv(os.path.join(minerals_dir, f"{EXPERIMENT_NAME}_{name}.csv"), index=False)
        plots.plot_individual_uncertainty(
            name, pts, MODEL_LABEL, N_BOOTSTRAP, CATEGORY,
            actual=None, comparison=baj.get(name),
            comparison_label="Barjamovic et al. (gravity model)",
            barjamovic_2011=baj_2011.get(name), forlanini_2008=forlanini.get(name),
            known_sites=known_sites, minerals=minerals, geo_screened_polygon=polygon, save_dir=fig_dir,
        )
    plots.plot_aggregate_map(
        lost_names, raw_results, MODEL_LABEL, CATEGORY, actual_coords_by_name=None,
        reference_coords_by_name=known_ref, reference_label="Known city",
        save_dir=fig_dir,
    )
    print(f"Figures saved under: {fig_dir}")

    return summary_df


if __name__ == "__main__":
    main()
