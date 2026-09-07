# -*- coding: utf-8 -*-
"""
Prediction for the 10 actually-lost ancient cities, MDS + Dyadic-share
weights, full 25-city similarity matrix, aligned every replicate using all
15 known cities as anchors (paper Table 8 -- the headline MDS lost-city
result).

Corresponds to legacy/MDSDyadicResults.py. This is the single most-affected
script in the MDS revision -- see CHANGELOG_AND_HANDOFF.md's MDS section for
the full writeup, summarized here:

Bug M1 fix: the legacy script built its similarity matrix as
S = matrix / col_sums (destination city's total-imports share only) then
symmetrized -- not the paper's own Eq. 2-3. Now uses
src/weights.py:dyadic_weights() on the symmetrized volume
(src/data.py:ancient_symmetric_matrix()), the same formula every LOO script
already used.

Bug M2 fix: the legacy script's Procrustes alignment had NO weight matrix
at all (A = (Y_known-c_y).T @ (X_known-c_x)) -- despite the paper's own
Sec. 2.1.2 stating explicitly that "the same weighting and alignment
procedure is applied" for the final prediction as for LOO. Now uses
src/mds.py:weighted_procrustes() with the same trade-derived weight vector
every LOO script uses.

Bug M3 fix: the legacy script drew 200 BOOTSTRAP_SEEDS from a MASTER_SEED
but its np.random.poisson(matrix_data) call ignored `seed` entirely,
consuming the global NumPy stream instead -- only MDS(random_state=seed)
actually used the per-replicate seed. Now goes through
src/train.py:resample_poisson(seed), independently seeded per replicate,
exactly like every other script in this package.

N_BOOTSTRAP=200, unchanged from the legacy script and matching GCN's
ancient_lost_dyadic.py.

No TII / no directional-constraint variant of this script: dyadic share
already outperformed TII in MDS's own LOO training (paper Sec 3.1/3.2), so
-- exactly as with the legacy scripts -- only dyadic is used for the actual
lost-city prediction; and directional constraints (Sec 7.1) were decided
out of scope for MDS (its closed-form Procrustes has no loss term to
penalize the way GCN's gradient descent does -- see CHANGELOG_AND_HANDOFF.md).

Reports distance to three independent estimates (Bug M1/M2 fixes make this
possible for free via evaluate.summarize_bootstrap()'s `comparisons` arg --
the legacy script only ever reported Dist_to_Baj):
  - Barjamovic et al.'s structural gravity model (Dist_to_Baj).
  - Barjamovic's own 2011 historical-geography monograph (Dist_to_Barjamovic2011).
  - Forlanini's 2008 historical-geography proposal (Dist_to_Forlanini2008).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from src import config, data, evaluate, plots, terrain  # noqa: E402

EXPERIMENT_NAME = "mds_ancient_lost_dyadic"
MODEL_LABEL = "MDS-Dyadic"
CATEGORY = "Lost"
N_BOOTSTRAP = 200
ELLIPSE_N_STD = 2.0  # matches the 2-sigma ellipse drawn on the plots


def main():
    summary_df, raw_results = evaluate.run_ancient_lost_prediction_mds(N_BOOTSTRAP)

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
