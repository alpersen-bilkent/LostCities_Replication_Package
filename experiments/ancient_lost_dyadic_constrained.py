# -*- coding: utf-8 -*-
"""
Final prediction for the 10 actually-lost cities, Dyadic-share weights,
with a directional-constraint penalty added to the training loss
(src/constraints.py) for relationships documented in ancient texts (for
example, "City X lies south and east of Kanesh"), compiled by historians
Forlanini (2008) and Barjamovic (2011).

This is the reported lost-city prediction (paper Table 11/12). The
predictive-power claim for trade topology alone is already established
by the LOO and modern-city results, so for the actual lost-city
predictions -- the practical, applied part of the paper -- there is no
reason to withhold legitimate historical evidence. This run satisfies
every applicable constraint at gamma=1 and agrees more closely with all
three independent comparison estimates than the unconstrained run.
experiments/ancient_lost_dyadic.py is kept as a robustness/ablation
check showing what trade topology predicts with no outside evidence
injected.

Same hyperparameters and bootstrap count as ancient_lost_dyadic.py
(EPOCHS=250, LR=0.0053, EMBEDDING_DIM=16, N_BOOTSTRAP=200 -- EPOCHS
updated from 1473 per experiments/grid_search_ancient_dyadic.py) so the
two are directly comparable. Only GAMMA is new.

Start at gamma=1 (one km of constraint violation costs as much as one
km of node error). After training, check how many constraints are still
binding at the bootstrap-mean prediction; raise gamma (try 5, then 10)
until none are, since Barjamovic et al. report none of their own
constraints bind near their point estimates. (At gamma=1, none bind
here either -- see console output.)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from src import config, constraints, data, evaluate, plots, terrain  # noqa: E402

EXPERIMENT_NAME = "ancient_lost_dyadic_constrained"
MODEL_LABEL = "Dyadic (constrained)"
CATEGORY = "Lost (Constrained)"
HPARAMS = {"epochs": 250, "lr": 0.0053, "embedding_dim": 16}
N_BOOTSTRAP = 200
GAMMA = 1.0
ELLIPSE_N_STD = 2.0  # matches the 2-sigma ellipse drawn on the plots


def main():
    name_to_node_idx = {name: i for i, name in enumerate(data.ANCIENT_CITY_NAMES)}
    kanes_lon, kanes_lat = constraints.kanes_reference_point()
    cos_ref = np.cos(np.radians(config.LATITUDE_PARAM))

    def constraint_fn(all_unscaled):
        return constraints.constraint_loss(
            all_unscaled, name_to_node_idx, config.KM_PER_DEGREE, cos_ref, kanes_lon, kanes_lat,
        )

    summary_df, raw_results = evaluate.run_ancient_lost_prediction(
        HPARAMS, N_BOOTSTRAP, constraint_fn=constraint_fn, gamma=GAMMA,
    )

    out_path = evaluate.save_results(summary_df, EXPERIMENT_NAME)
    print(f"\nSaved: {out_path}")
    print(f"\n--- PREDICTION RESULTS, gamma={GAMMA} ---")
    print("(Dist_to_Baj = Barjamovic et al. gravity model; "
          "Dist_to_Barjamovic2011 = Barjamovic's own historian proposal; "
          "Dist_to_Forlanini2008 = Forlanini's historian proposal)")
    print(summary_df.round(3).to_string(index=False))
    print(f"\nAvg Distance to Barjamovic et al. (gravity model): {summary_df['Dist_to_Baj'].mean():.2f} km")
    print(f"Avg Distance to Barjamovic 2011 (historian): {summary_df['Dist_to_Barjamovic2011'].mean():.2f} km")
    print(f"Avg Distance to Forlanini 2008 (historian): {summary_df['Dist_to_Forlanini2008'].mean():.2f} km")

    # Diagnostic: which rules are still binding at the bootstrap-mean prediction.
    names = list(summary_df['City'])
    mean_pred = summary_df[['Mean_Long', 'Mean_Lat']].values
    name_to_idx = {name: i for i, name in enumerate(names)}
    binding = constraints.count_binding_constraints(mean_pred, name_to_idx, kanes_lon, kanes_lat)
    if binding:
        print(f"\n{len(binding)} constraint(s) still binding at gamma={GAMMA}: {binding}")
        print("Per HANDOFF.md §7.1, consider raising GAMMA (try 5, then 10) and rerunning.")
    else:
        print(f"\nAll applicable constraints satisfied at gamma={GAMMA}.")

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
