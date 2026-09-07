# -*- coding: utf-8 -*-
"""
15 modern Turkish province centers, LOO, MDS + TII weights (paper Table 14).
Bug M4 does not apply here (no lost cities).

Uses src/weights.py:symmetric_tii_from_symmetric() -- the RAW, pre-log-norm
TII score -- not tii_weights_from_symmetric() (GCN's Eq. 9 transform); see
mds_ancient_loo_tii.py's docstring for why.

Corresponds to legacy/MDSRealTIIBootstrap.py (again, "Real" here means
modern-city data, not a final-prediction script). N_BOOTSTRAP=40, matching
GCN's modern_loo_tii.py.

Bug M3 fix: resampling goes through src/train.py:resample_poisson(seed),
independently seeded per replicate.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from src import config, data, evaluate, plots  # noqa: E402

EXPERIMENT_NAME = "mds_modern_loo_tii"
MODEL_LABEL = "MDS-TII"
CATEGORY = "Modern"
N_BOOTSTRAP = 40


def main():
    summary_df, raw_results = evaluate.run_modern_loo_mds("tii", N_BOOTSTRAP)

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
