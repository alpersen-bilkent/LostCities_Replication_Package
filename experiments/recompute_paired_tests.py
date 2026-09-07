# -*- coding: utf-8 -*-
"""
One-off utility: computes the new paired Wilcoxon test (added after the
four LOO scripts below were already run) directly from their EXISTING
saved results, instead of rerunning model training/bootstrapping.

Safe to do because the paired test only needs each city's already-
computed Error_KM (results/{name}.parquet's own City/Error_KM columns,
unchanged by the addition of this test) and Barjamovic et al.'s own
per-city known-city errors (fixed data, evaluate.barjamovic_known_city_
errors()) -- nothing this script reads depends on anything that changed
today. Run the four LOO scripts themselves (ancient_loo_dyadic.py,
ancient_loo_tii.py, mds_ancient_loo_dyadic.py, mds_ancient_loo_tii.py)
again ONLY if their results/*.parquet files don't already exist or are
stale for some other reason -- this script errors clearly if a required
file is missing, rather than silently skipping it.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from src import config, data, evaluate  # noqa: E402

EXPERIMENT_NAMES = [
    "ancient_loo_dyadic",
    "ancient_loo_tii",
    "mds_ancient_loo_dyadic",
    "mds_ancient_loo_tii",
]


def main():
    _, _, known_names, _ = data.ancient_known_lost_split()
    baj_known_errors = evaluate.barjamovic_known_city_errors()

    for experiment_name in EXPERIMENT_NAMES:
        results_path = os.path.join(config.RESULTS_DIR, f"{experiment_name}.parquet")
        if not os.path.exists(results_path):
            raise FileNotFoundError(
                f"{results_path} not found -- run experiments/{experiment_name}.py first "
                f"(this script only recomputes the paired test from an EXISTING result, "
                f"it doesn't generate one)."
            )
        summary_df = pd.read_parquet(results_path)
        point_errors = dict(zip(summary_df['City'], summary_df['Error_KM']))

        missing = [name for name in known_names if name not in point_errors]
        if missing:
            raise ValueError(
                f"{results_path} is missing Error_KM for: {missing} -- this result may "
                f"predate a change to which cities are scored; rerun experiments/"
                f"{experiment_name}.py rather than trust a partial paired test."
            )

        paired_sig = evaluate.paired_significance_test_vs_reference(
            point_errors, baj_known_errors, known_names,
        )
        paired_sig_path = evaluate.save_significance(paired_sig, f"{experiment_name}_paired")

        print(f"\n--- {experiment_name}: PAIRED (CITY-MATCHED) WILCOXON TEST vs. "
              f"Barjamovic et al. (2019) ({paired_sig['n_cities_paired']} known cities) ---")
        print(f"Mean paired difference (this method minus Barjamovic, per city): "
              f"{paired_sig['mean_paired_diff_km']:.2f} km")
        print(f"Wilcoxon signed-rank: statistic={paired_sig['wilcoxon_statistic']:.2f}, "
              f"p={paired_sig['wilcoxon_p_value_one_sided']:.4f}")
        print(f"Saved: {paired_sig_path}")


if __name__ == "__main__":
    main()
