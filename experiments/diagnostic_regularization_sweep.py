# -*- coding: utf-8 -*-
"""
DIAGNOSTIC, not a reported result. Investigates whether the unconstrained
ancient lost-city predictions' visible collapse toward the network's
centroid (and apparent near-collinearity along one axis) is sensitive to
two hyperparameters that were never actually tuned anywhere in this
codebase -- weight_decay (hardcoded at 5e-4 in src/train.py, inherited
unchanged from the legacy scripts) and embedding_dim (grid_search_
ancient_dyadic.py's EMBEDDING_CHOICES = [16] only ever tested one value,
also just the legacy default).

This script does NOT change anything in the reported pipeline. Both
parameters were added to src/train.py and src/evaluate.py as optional,
default-preserving arguments specifically so this sweep can override
them without touching any already-reported script -- see
train.fit_predict()'s weight_decay parameter and the same parameter
threaded through evaluate.run_ancient_loo() /
evaluate.run_ancient_lost_prediction().

Deliberately reduced N_BOOTSTRAP (15, not 200) and a small grid: this is
a screening pass to see whether there is any signal worth a full,
properly-powered follow-up, not a publication-grade result in itself.
Every number below should be read with that caveat.

Three things are reported per (weight_decay, embedding_dim) combination,
because "spread the lost-city predictions out more" is not on its own
evidence of anything better -- it would be exactly the kind of arbitrary
fix that should NOT be adopted just because it looks less centralized:

  1. Known-city LOO error (km) -- the ONLY quantity in this whole problem
     that can be checked against ground truth. A combination that
     degrades this is a combination that generalizes worse, full stop,
     regardless of how its lost-city map looks.
  2. Collinearity ratio of the 10 lost-city point estimates -- the
     smaller eigenvalue divided by the larger eigenvalue of their
     covariance (computed in km-space, using the same lon/lat -> km
     conversion as the rest of this codebase, config.KM_PER_DEGREE /
     config.LATITUDE_PARAM, so the ratio isn't distorted by longitude
     and latitude having different km-per-degree). 0 = perfectly
     collinear (all 10 points on one line); 1 = perfectly isotropic
     (spread evenly in every direction). This is a direct, standard
     (PCA) measure of exactly the "strung out in a line" pattern
     observed on the aggregate map -- not a proxy or a guess.
  3. Average distance to the three independent comparison estimates
     (Barjamovic et al.'s gravity model, Barjamovic 2011, Forlanini
     2008) -- if a combination increases spread (2) while moving further
     from all three independent estimates, that is evidence the spread
     is noise, not signal. Only a combination that helps or holds (1)
     AND increases (2) AND does not worsen (3) would be a genuine
     candidate for adoption -- and even then, only after a properly
     powered confirmation run and a decision applied consistently across
     related scripts, exactly like the N_BOOTSTRAP=200 decision earlier
     in this project.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config, data, evaluate  # noqa: E402

N_BOOTSTRAP_DIAGNOSTIC = 15  # reduced -- screening pass, not a final result
EPOCHS = 250  # held fixed at the current pipeline's value; isolate wd/embedding_dim only
LR = 0.0053

WEIGHT_DECAY_GRID = [5e-4, 1e-4, 0.0]   # 5e-4 = current/baseline (unchanged everywhere else)
EMBEDDING_DIM_GRID = [16, 32]            # 16 = current/baseline


def collinearity_ratio(points_lonlat):
    """Smaller/larger eigenvalue of the covariance of `points_lonlat`
    (N, 2) after converting to an approximately equal-area km-space
    (same conversion used throughout this codebase) -- 0 = collinear,
    1 = isotropic spread. A direct PCA measure, not a heuristic."""
    cos_ref = np.cos(np.radians(config.LATITUDE_PARAM))
    x_km = points_lonlat[:, 0] * cos_ref * config.KM_PER_DEGREE
    y_km = points_lonlat[:, 1] * config.KM_PER_DEGREE
    xy_km = np.column_stack([x_km, y_km])
    cov = np.cov(xy_km, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)  # ascending
    eigvals = np.clip(eigvals, 0, None)
    if eigvals[-1] <= 1e-9:
        return float('nan')
    return float(eigvals[0] / eigvals[-1])


def main():
    known_idx, lost_idx, known_names, lost_names = data.ancient_known_lost_split()
    known_coords, ordered_known_names = data.ancient_known_coords()
    actual = {name: known_coords[i] for i, name in enumerate(ordered_known_names)}
    baj = data.barjamovic_lost_lookup()
    baj_2011 = data.barjamovic_2011_lookup()
    forlanini = data.forlanini_2008_lookup()

    rows = []
    for wd in WEIGHT_DECAY_GRID:
        for emb in EMBEDDING_DIM_GRID:
            hp = {"embedding_dim": emb, "epochs": EPOCHS, "lr": LR}
            tag = f"wd={wd:g}, emb={emb}"
            print(f"\n{'=' * 70}\n{tag}\n{'=' * 70}")

            print("-- known-city LOO (ground-truth-anchored check) --")
            loo_summary, _ = evaluate.run_ancient_loo(
                'dyadic', hp, N_BOOTSTRAP_DIAGNOSTIC, weight_decay=wd,
            )
            known_loo_error = float(loo_summary['Error_KM'].mean())

            print("-- lost-city prediction (unconstrained) --")
            lost_summary, raw_results = evaluate.run_ancient_lost_prediction(
                hp, N_BOOTSTRAP_DIAGNOSTIC, weight_decay=wd,
            )
            mean_points = np.array([
                np.array(raw_results[name]).mean(axis=0) for name in lost_names
            ])
            collin = collinearity_ratio(mean_points)
            avg_dist_baj = float(lost_summary['Dist_to_Baj'].mean())
            avg_dist_b2011 = float(lost_summary['Dist_to_Barjamovic2011'].mean())
            avg_dist_forl = float(lost_summary['Dist_to_Forlanini2008'].mean())

            rows.append({
                'weight_decay': wd, 'embedding_dim': emb,
                'Known_LOO_Error_km': known_loo_error,
                'Collinearity_Ratio': collin,
                'Avg_Dist_to_Baj_km': avg_dist_baj,
                'Avg_Dist_to_Barjamovic2011_km': avg_dist_b2011,
                'Avg_Dist_to_Forlanini2008_km': avg_dist_forl,
            })

    report = pd.DataFrame(rows)
    print(f"\n\n{'=' * 90}\nFULL SCREENING REPORT (N_BOOTSTRAP={N_BOOTSTRAP_DIAGNOSTIC}, "
          f"epochs={EPOCHS} -- read with that caveat)\n{'=' * 90}")
    print(report.round(3).to_string(index=False))

    baseline = report[(report['weight_decay'] == 5e-4) & (report['embedding_dim'] == 16)].iloc[0]
    print(f"\nBaseline (current pipeline: wd=5e-4, emb=16): "
          f"Known_LOO_Error={baseline['Known_LOO_Error_km']:.2f} km, "
          f"Collinearity_Ratio={baseline['Collinearity_Ratio']:.3f}")
    print("\nA candidate is only worth pursuing further if, relative to baseline, it does NOT "
          "increase Known_LOO_Error_km, DOES increase Collinearity_Ratio (less collinear), and "
          "does NOT increase all three Avg_Dist_to_* columns. Anything else is not a genuine "
          "improvement, however different the map looks.")

    out_dir = os.path.join(config.RESULTS_DIR, "diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "regularization_sweep.csv")
    report.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path} (diagnostic only -- not part of the reported results/ tables)")

    return report


if __name__ == "__main__":
    main()
