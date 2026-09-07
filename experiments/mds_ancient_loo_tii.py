# -*- coding: utf-8 -*-
"""
15 fictitiously-held-out known ancient cities, LOO, MDS + TII weights
(paper Table 7). SMACOF embedding + weighted Procrustes (src/mds.py),
always over the full 25-city similarity matrix (Bug M4 fix).

Uses src/weights.py:symmetric_tii_from_directed() -- the RAW, pre-log-norm
TII score (paper Eq. 5-6) -- not tii_weights_from_directed(), which applies
GCN's Eq. 9 log+min-max transform. MDS's own TII dissimilarity (Eq. 7) has
no such step; see src/weights.py and tests/test_mds.py for why using the
GCN-ready weights here would be silently wrong.

Corresponds to legacy/MDSTIIBootstrap.py. N_BOOTSTRAP raised 40 -> 200 to
match GCN's ancient_loo_tii.py, for the same significance-test-precision
reason as mds_ancient_loo_dyadic.py.

Bug M3 fix: resampling goes through src/train.py:resample_poisson(seed),
independently seeded per replicate.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from src import config, data, evaluate, plots  # noqa: E402

EXPERIMENT_NAME = "mds_ancient_loo_tii"
MODEL_LABEL = "MDS-TII"
CATEGORY = "LOO"
N_BOOTSTRAP = 200


def main():
    summary_df, raw_results = evaluate.run_ancient_loo_mds("tii", N_BOOTSTRAP)

    out_path = evaluate.save_results(summary_df, EXPERIMENT_NAME)
    print(f"\nSaved: {out_path}")
    print(f"\n--- {EXPERIMENT_NAME.upper()} RESULTS ---")
    print(summary_df.round(3).to_string(index=False))
    print(f"\nOverall Mean Error: {summary_df['Error_KM'].mean():.2f} km")

    known_coords, known_names = data.ancient_known_coords()
    actual = {name: known_coords[i] for i, name in enumerate(known_names)}
    baj = data.barjamovic_known_lookup()

    resampled_errors = evaluate.bootstrap_ensemble_mean_distribution(raw_results, known_names, actual)
    sig = evaluate.significance_test_vs_reference(
        resampled_errors, config.BARJAMOVIC_LOO_HEADLINE_KM, "Barjamovic et al. (2019)",
    )
    sig_path = evaluate.save_significance(sig, EXPERIMENT_NAME)
    print(f"\n--- SIGNIFICANCE TEST vs. {sig['reference_label']} ({sig['reference_km']:.2f} km) ---")
    print(f"Resampled ensemble-mean error: {sig['bootstrap_mean_km']:.2f} km "
          f"(95% CI: {sig['ci_95_low_km']:.2f}-{sig['ci_95_high_km']:.2f} km, "
          f"n={sig['n_resamples']} resamples)")
    print(f"Fraction of resamples NOT better than the reference: {sig['p_value_one_sided']:.3f}")
    print(f"Saved: {sig_path}")

    point_errors = dict(zip(summary_df['City'], summary_df['Error_KM']))
    baj_known_errors = evaluate.barjamovic_known_city_errors()
    paired_sig = evaluate.paired_significance_test_vs_reference(
        point_errors, baj_known_errors, known_names,
    )
    paired_sig_path = evaluate.save_significance(paired_sig, f"{EXPERIMENT_NAME}_paired")
    print(f"\n--- PAIRED (CITY-MATCHED) WILCOXON TEST vs. Barjamovic et al. (2019) "
          f"({paired_sig['n_cities_paired']} known cities) ---")
    print("(Wilcoxon signed-rank, one-sided, exact method -- this method's own per-city "
          "error vs. Barjamovic's per-city error, matched city for city; see "
          "evaluate.paired_significance_test_vs_reference()'s docstring)")
    print(f"Mean paired difference (this method minus Barjamovic, per city): "
          f"{paired_sig['mean_paired_diff_km']:.2f} km")
    print(f"Wilcoxon signed-rank: statistic={paired_sig['wilcoxon_statistic']:.2f}, "
          f"p={paired_sig['wilcoxon_p_value_one_sided']:.4f}")
    print(f"Saved: {paired_sig_path}")

    fig_dir = os.path.join(config.FIGURES_DIR, EXPERIMENT_NAME)
    for name in known_names:
        plots.plot_individual_uncertainty(
            name, np.array(raw_results[name]), MODEL_LABEL, N_BOOTSTRAP, CATEGORY,
            actual=actual[name], comparison=baj.get(name), comparison_label="Barjamovic",
            save_dir=fig_dir,
        )
    plots.plot_aggregate_map(
        known_names, raw_results, MODEL_LABEL, CATEGORY, actual_coords_by_name=actual,
        overall_error_km=summary_df['Error_KM'].mean(),
        save_dir=fig_dir,
    )
    print(f"Figures saved under: {fig_dir}")

    return summary_df


if __name__ == "__main__":
    main()
