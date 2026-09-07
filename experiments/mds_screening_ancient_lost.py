# -*- coding: utf-8 -*-
"""
Sec. 7.3 geographic screening applied to MDS's lost-city predictions --
parity with GCN's screening_ancient_lost.py, reusing src/terrain.py
unchanged (nothing in it is GCN-specific; it takes any
(mean_long, mean_lat, std_long, std_lat), regardless of which method
produced them).

Reads results/mds_ancient_lost_dyadic.parquet by default (run
experiments/mds_ancient_lost_dyadic.py first). See that script's own
docstring, and screening_ancient_lost.py's docstring, for the full
rationale -- identical here, just pointed at MDS's output instead of GCN's.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from src import config, data, plots, terrain  # noqa: E402

RESULTS_NAME = "mds_ancient_lost_dyadic"
MODEL_LABEL = "MDS-Dyadic"
CATEGORY = "Screening"
N_STD = 2.0
N_PER_AXIS = 25  # was 15 -- see screening_ancient_lost.py's matching comment
MAX_SLOPE_DEG = None
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
    fig_dir = os.path.join(config.FIGURES_DIR, "mds_screening_ancient_lost")
    os.makedirs(out_dir, exist_ok=True)

    for city, df in screened.items():
        out_path = os.path.join(out_dir, f"mds_{city}.csv")
        df.drop(columns=['known_site']).to_csv(out_path, index=False)
        print(f"\n--- {city}: top {TOP_N} candidates ({len(df)} scored) ---")
        print(df.drop(columns=['known_site']).head(TOP_N).round(3).to_string(index=False))

        known_in_range = df[df['known_site'].notna()] if 'known_site' in df.columns else df.iloc[0:0]
        if len(known_in_range) > 0:
            n_scored = len(df)
            rank_positions = df['rank_score'].rank(method='min')
            print(f"Known site(s) inside this ellipse, ranked against all {n_scored} candidates:")
            for _, row in known_in_range.sort_values('rank_score', na_position='last').iterrows():
                # Bug fix (found via mds_ancient_lost_dyadic.py's wider ellipses --
                # e.g. Suppiluliya's reaches the Black Sea coast, where
                # terrain.sample_slope()'s neighbor-point gradient sampling can
                # land outside the elevation raster and return NaN even though
                # the site's own elevation is valid). A NaN rank_score means
                # "couldn't be scored, missing terrain data at this location" --
                # report that plainly instead of crashing on int(NaN).
                raw_position = rank_positions[row.name]
                if pd.isna(raw_position):
                    print(f"    {row['known_site']}: rank unavailable (missing terrain data at this "
                          f"location) of {n_scored} "
                          f"(distance to prediction: {row['dist_to_prediction_km']:.1f} km)")
                else:
                    print(f"    {row['known_site']}: rank {int(raw_position)} of {n_scored} "
                          f"(distance to prediction: {row['dist_to_prediction_km']:.1f} km)")

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
