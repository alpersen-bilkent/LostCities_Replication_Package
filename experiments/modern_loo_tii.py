# -*- coding: utf-8 -*-
"""
15 modern Turkish province centers, LOO, TII weights.
Bug 2 does not apply here: there are no lost cities, so the graph was
already exactly the known set in the original script.

Corresponds to legacy/GCNTIIRealBootstrap.py. EMBEDDING_DIM=16 unchanged.
LR=0.0051, EPOCHS=2390 updated from 0.005/596 per
experiments/grid_search_modern_tii.py. N_BOOTSTRAP=40 -- kept lower than
the ancient LOO scripts' 200 since EPOCHS here is much higher (2390 vs.
62), and no significance test depends on this script's replicate count
the way it does for the ancient LOO scripts, so the added precision from
a larger B isn't needed enough to justify the extra runtime.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from src import config, data, evaluate, plots  # noqa: E402

EXPERIMENT_NAME = "modern_loo_tii"
MODEL_LABEL = "TII"
CATEGORY = "Modern"
HPARAMS = {"epochs": 2390, "lr": 0.0051, "embedding_dim": 16}
N_BOOTSTRAP = 40


def main():
    summary_df, raw_results = evaluate.run_modern_loo("tii", HPARAMS, N_BOOTSTRAP)

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
