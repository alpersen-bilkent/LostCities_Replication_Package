# -*- coding: utf-8 -*-
"""
One-off utility: recomputes Dist_to_Baj in every saved lost-city results
file after correcting src/data.py:BARJAMOVIC_LOST_ESTIMATES from the NBER
working-paper draft's Appendix Table F.1 to the published QJE paper's own
Table II (see that constant's docstring for the full citation trail and
verification).

Safe to do without retraining: Dist_to_Baj is just the great-circle-ish
distance (train.calculate_distance_km()) between each city's ALREADY-
COMPUTED Mean_Lat/Mean_Long (the bootstrap-mean prediction, unaffected by
this fix) and Barjamovic et al.'s reference point (the only thing that
changed). Dist_to_Barjamovic2011 and Dist_to_Forlanini2008 are untouched --
those reference historians' own separate proposals, not Table II.

Run this AFTER confirming src/data.py's BARJAMOVIC_LOST_ESTIMATES has
already been corrected; this script does not itself change that constant,
only re-derives distances from it.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from src import config, data, train  # noqa: E402

RESULTS_NAMES = [
    "ancient_lost_dyadic",               # GCN, unconstrained (ablation)
    "ancient_lost_dyadic_constrained",   # GCN, constrained (primary reported result)
    "mds_ancient_lost_dyadic",           # MDS
]


def main():
    baj = data.barjamovic_lost_lookup()  # name -> (long, lat), Table II values now

    for name in RESULTS_NAMES:
        path = os.path.join(config.RESULTS_DIR, f"{name}.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found -- run experiments/{name}.py first.")
        df = pd.read_parquet(path)

        old_mean = df['Dist_to_Baj'].mean()
        new_vals = []
        for _, row in df.iterrows():
            city = row['City']
            comp_long, comp_lat = baj[city]
            new_vals.append(train.calculate_distance_km(
                row['Mean_Long'], row['Mean_Lat'], comp_long, comp_lat,
            ))
        df['Dist_to_Baj'] = new_vals
        new_mean = df['Dist_to_Baj'].mean()

        df.to_parquet(path, index=False)

        print(f"\n--- {name} ---")
        print(df[['City', 'Dist_to_Baj']].round(2).to_string(index=False))
        print(f"Mean Dist_to_Baj: {old_mean:.2f} km (old, F.1-based) -> "
              f"{new_mean:.2f} km (new, Table II-based)")
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
