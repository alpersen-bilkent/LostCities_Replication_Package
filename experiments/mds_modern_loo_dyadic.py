# -*- coding: utf-8 -*-
"""
15 modern Turkish province centers, LOO, MDS + Dyadic-share weights
(paper Table 13). Bug M4 does not apply here: there are no lost cities, so
the embedded set is already exactly the known set.

Corresponds to legacy/MDSDyadicRealBootstrap.py (despite the "Real" in its
name, that script is the modern-city LOO script, not a final-prediction
script -- see CHANGELOG_AND_HANDOFF.md's MDS section for the naming note).
N_BOOTSTRAP=40, matching GCN's modern_loo_dyadic.py (no significance test
depends on this script's replicate count).

Bug M3 fix: resampling goes through src/train.py:resample_poisson(seed),
independently seeded per replicate.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from src import config, data, evaluate, plots  # noqa: E402

EXPERIMENT_NAME = "mds_modern_loo_dyadic"
MODEL_LABEL = "MDS-Dyadic"
CATEGORY = "Modern"
N_BOOTSTRAP = 40


def main():
    summary_df, raw_results = evaluate.run_modern_loo_mds("dyadic", N_BOOTSTRAP)

    out_path = evaluate.save_results(summary_df, EXPERIMENT_NAME)
    print(f"\nSaved: {out_path}")
    print(f"\n--- {EXPERIMENT_NAME.upper()} RESULTS ---")
    print(summary_df.round(3).to_string(index=False))
    print(f"\nOverall Mean Error: {summary_df['Error_KM'].mean():.2f} km")

    valid_cities, known_coords = data.modern_cities_and_coords()
    actual = {name: known_coords[i] for i, name in enumerate(valid_cities)}

    fig_dir = os.path.join(config.FIGURES_DIR, EXPERIMENT_NAME)
    for name in valid_cities:
        plots.plot_individual_uncertainty(
            name, np.array(raw_results[name]), MODEL_LABEL, N_BOOTSTRAP, CATEGORY,
            actual=actual[name], save_dir=fig_dir,
        )
    plots.plot_aggregate_map(
        valid_cities, raw_results, MODEL_LABEL, CATEGORY, actual_coords_by_name=actual,
        overall_error_km=summary_df['Error_KM'].mean(),
        save_dir=fig_dir,
        xlim=(31, 39.3), ylim=(35.8, 42.2),
    )
    print(f"Figures saved under: {fig_dir}")

    return summary_df


if __name__ == "__main__":
    main()
