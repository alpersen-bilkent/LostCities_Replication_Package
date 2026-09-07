# -*- coding: utf-8 -*-
"""
§7.3: rank candidate survey sites inside each lost city's 2-sigma
confidence ellipse, using src/terrain.py (elevation/slope, distance to
water, and distance to the nearest known archaeological site from
data.KNOWN_ARCHAEOLOGICAL_SITES -- all from Barjamovic et al.'s own
replication package / supplemental appendix).

The known-site criterion is deliberately city-agnostic: it does not
pair specific sites with specific lost cities the way Barjamovic et
al.'s own Appendix Table 3 does. Reusing their city-to-site pairing to
"check" our model's ellipse would be circular -- it would just measure
agreement with a conclusion we imported, not do independent screening.

This script's grid-cell ranking uses the axis-aligned ellipse
approximation (mean/std only -- see terrain.screen_ellipse()), because
it only reads the saved summary table, not the raw bootstrap points. For
the more precise, non-elliptical geography-carved region (Mahalanobis
ellipse + hard terrain/water exclusions, no invented weights -- see
terrain.geo_screened_region()) and its own "known sites/minerals inside"
tables, see results/known_sites/ and results/minerals/, written directly
by experiments/ancient_lost_dyadic_constrained.py, which has the raw
points in memory. That's the authoritative "which höyüks are candidates"
table; this script's {city}.csv is a complementary, finer-grained
ranking of individual grid cells by terrain quality within the broader
region.

INCLUDE_KNOWN_SITES=True (default) folds any known archaeological site
inside a city's ellipse into the same ranked pool as the grid cells, so
the output/plot show directly whether a real, named site happens to
land in a well-ranked spot -- an independent check, since the ranking
criteria (slope, water, clustering with other known sites) never used
that site's identity or its distance to any particular lost city.

Reads results/ancient_lost_dyadic_constrained.parquet by default -- the
reported lost-city prediction (run
experiments/ancient_lost_dyadic_constrained.py first). Set RESULTS_NAME
below to "ancient_lost_dyadic" to screen the unconstrained ablation
instead.
"""

# Type "import os; os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' in the kernel" before running if crashes
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from src import config, data, plots, terrain  # noqa: E402

RESULTS_NAME = "ancient_lost_dyadic_constrained"
MODEL_LABEL = "Dyadic"
CATEGORY = "Screening"
N_STD = 2.0
N_PER_AXIS = 25  # was 15 -- denser grid makes the rank_score surface's interior
                 # smoother; the plotted BOUNDARY comes from the exact ellipse
                 # (plots.plot_screening_ranking's ellipse_cov clip), not from
                 # this grid's own edge, so this only affects interior smoothness.
MAX_SLOPE_DEG = None  # e.g. 15.0 to exclude steep/high-Taurus cells; unset for now
TOP_N = 10
INCLUDE_KNOWN_SITES = True


def main():
    results_path = os.path.join(config.RESULTS_DIR, f"{RESULTS_NAME}.parquet")
    if not os.path.exists(results_path):
        raise FileNotFoundError(
            f"{results_path} not found -- run experiments/{RESULTS_NAME}.py first."
        )
    summary_df = pd.read_parquet(results_path)

    baj = data.barjamovic_lost_lookup()
    baj_2011 = data.barjamovic_2011_lookup()
    forlanini = data.forlanini_2008_lookup()

    screened = terrain.screen_all_lost_cities(
        summary_df, n_std=N_STD, n_per_axis=N_PER_AXIS, max_slope_deg=MAX_SLOPE_DEG,
        include_known_sites=INCLUDE_KNOWN_SITES,
    )

    out_dir = os.path.join(config.RESULTS_DIR, "screening")
    fig_dir = os.path.join(config.FIGURES_DIR, "screening_ancient_lost")
    os.makedirs(out_dir, exist_ok=True)

    for city, df in screened.items():
        out_path = os.path.join(out_dir, f"{city}.csv")
        df.drop(columns=['known_site']).to_csv(out_path, index=False)
        print(f"\n--- {city}: top {TOP_N} candidates ({len(df)} scored) ---")
        print(df.drop(columns=['known_site']).head(TOP_N).round(3).to_string(index=False))

        known_in_range = df[df['known_site'].notna()] if 'known_site' in df.columns else df.iloc[0:0]
        if len(known_in_range) > 0:
            n_scored = len(df)
            rank_positions = df['rank_score'].rank(method='min')
            print(f"Known site(s) inside this ellipse, ranked against all {n_scored} candidates:")
            for _, row in known_in_range.sort_values('rank_score', na_position='last').iterrows():
                # Bug fix (found via the MDS screening script's wider ellipses,
                # not previously triggered by any GCN lost-city ellipse -- see
                # CHANGELOG_AND_HANDOFF.md's MDS section, Bug M7). A NaN
                # rank_score means terrain.sample_slope()'s gradient sampling
                # landed outside the elevation raster's coverage (can happen
                # near a coastline) even though the site's own elevation is
                # valid -- report that plainly instead of crashing on int(NaN).
                raw_position = rank_positions[row.name]
                if pd.isna(raw_position):
                    print(f"    {row['known_site']}: rank unavailable (missing terrain data at this "
                          f"location) of {n_scored} "
                          f"(distance to prediction: {row['dist_to_prediction_km']:.1f} km)")
                else:
                    print(f"    {row['known_site']}: rank {int(raw_position)} of {n_scored} "
                          f"(distance to prediction: {row['dist_to_prediction_km']:.1f} km)")

        # Informational overlay only -- never part of rank_score (see
        # terrain.py's module docstring: too little data / not a
        # significant predictor in Barjamovic et al.'s own regression).
        city_row = summary_df[summary_df['City'] == city].iloc[0]
        cov = None
        if {'Var_Long', 'Var_Lat', 'Cov_LonLat'}.issubset(summary_df.columns):
            cov = [[city_row['Var_Long'], city_row['Cov_LonLat']],
                   [city_row['Cov_LonLat'], city_row['Var_Lat']]]
        minerals = terrain.minerals_within_ellipse(
            city_row['Mean_Long'], city_row['Mean_Lat'], city_row['Std_Long'], city_row['Std_Lat'],
            n_std=N_STD, cov=cov,
        )

        plots.plot_screening_ranking(
            city, df, CATEGORY, model_label=MODEL_LABEL, save_dir=fig_dir,
            comparison=baj.get(city), comparison_label="Barjamovic et al. (gravity model)",
            barjamovic_2011=baj_2011.get(city), forlanini_2008=forlanini.get(city),
            gcn_mean=(city_row['Mean_Long'], city_row['Mean_Lat']),
            minerals=minerals, ellipse_cov=cov, ellipse_n_std=N_STD,
        )

    print(f"\nFull ranked candidate-cell tables saved under: {out_dir}")
    print(f"Ranking plots saved under: {fig_dir}")
    print(f"For the precise geography-carved region and its known-sites/minerals tables, "
          f"see {os.path.join(config.RESULTS_DIR, 'known_sites')} and "
          f"{os.path.join(config.RESULTS_DIR, 'minerals')} (written by {RESULTS_NAME}.py).")


if __name__ == "__main__":
    main()
