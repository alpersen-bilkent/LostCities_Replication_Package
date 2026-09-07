# -*- coding: utf-8 -*-
"""
LOO and bootstrap drivers. Each experiment script in ../experiments/
is just a hyperparameter dict plus a call into one of the functions
below, so all five original scripts collapse into this one place.

Bug 2 fix
---------
The original GCNDyadicBootstrap.py and GCNTIIBootstrap.py built the
training graph from `matrix[np.ix_(known_indices, known_indices)]` --
a 15-node graph with the 10 lost cities absent entirely. But paper
§2.2.3 argues the model is forced to learn a meaningful embedding for
e.g. Zalpa because it is a neighbour of Kanes and error backpropagates
through Zalpa's embedding when predicting Kanes -- a mechanism that
only exists if Zalpa is a node in the graph. run_ancient_loo() below
always builds the full 25-node graph and only masks which nodes
*supervise the loss*, never which nodes exist.
"""

import os

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from . import config, data, mds, train, weights


def run_loo_experiment(num_nodes, resample_fn, weight_fn, known_node_idx, known_coords,
                        known_names, hparams, n_bootstrap, master_seed=config.MASTER_SEED,
                        desc="Monte Carlo Iterations", weight_decay=5e-4):
    """Generic leave-one-out bootstrap over a set of supervised nodes sitting
    inside a graph of num_nodes total nodes (num_nodes may exceed
    len(known_node_idx), e.g. the ancient network's lost cities).

    Bug fix (found during a later review, not in HANDOFF.md's original
    list): this used to fit ONE MinMaxScaler on all known_coords before
    the fold loop, then reuse it for every fold -- including the fold
    where that same city is the one being held out. Since the scaler's
    min/max range (and hence the coordinate system used both to train
    the loss and to scaler.inverse_transform() that fold's own
    prediction) was calibrated using that city's own true coordinate,
    this let a small amount of information about the "unknown" answer
    leak into the very normalization used to train and decode it --
    a violation of train/test separation (preprocessing transforms
    should only ever be fit on the training portion). Fixed by fitting a
    separate scaler per fold, on only that fold's 14 training cities."""
    rng = np.random.RandomState(master_seed)
    bootstrap_seeds = rng.randint(0, 10000, size=n_bootstrap)

    known_node_idx = np.asarray(known_node_idx)
    num_known = len(known_node_idx)

    fold_scalers = []
    fold_targets_scaled = []
    for i in range(num_known):
        train_mask_local = np.array([j for j in range(num_known) if j != i])
        fold_scaler = MinMaxScaler().fit(known_coords[train_mask_local])
        fold_scalers.append(fold_scaler)
        fold_targets_scaled.append(
            torch.tensor(fold_scaler.transform(known_coords), dtype=torch.float)
        )

    results = {name: [] for name in known_names}

    for seed in tqdm(bootstrap_seeds, desc=desc):
        raw_matrix = resample_fn(seed)
        weight_matrix = weight_fn(raw_matrix)
        graph_data = data.create_graph_data(weight_matrix, num_nodes)

        for i in range(num_known):
            train_mask_local = np.array([j for j in range(num_known) if j != i])
            predict_idx = known_node_idx[i:i + 1]

            preds = train.fit_predict(
                graph_data, num_nodes, hparams['embedding_dim'], hparams['epochs'],
                hparams['lr'], seed, fold_targets_scaled[i], fold_scalers[i],
                known_node_idx, train_mask_local, predict_idx, weight_decay=weight_decay,
            )
            results[known_names[i]].append(preds[0])

    return results, fold_scalers


def summarize_bootstrap(results, names, actual_coords_by_name=None, comparisons=None):
    """comparisons: optional dict {column_name: {city_name: (lon, lat)}} --
    one Dist_to_<column_name> = ... column added per entry, so a table can
    report distance to several independent estimates at once (e.g. the
    GCN vs. Barjamovic et al.'s gravity model AND vs. two historians'
    proposals) without conflating them under one ambiguous column name."""
    comparisons = comparisons or {}
    rows = []
    for name in names:
        pts = np.array(results[name])
        mean_c = pts.mean(axis=0)
        std_c = pts.std(axis=0)
        row = {
            'City': name,
            'Mean_Lat': mean_c[1], 'Mean_Long': mean_c[0],
            'Std_Lat': std_c[1], 'Std_Long': std_c[0],
        }
        # Full 2x2 covariance of the bootstrap cloud (ddof=1, matching
        # np.cov's default -- the same convention plots.draw_confidence_
        # ellipse() and terrain.geo_screened_region() already use to draw
        # the rotated per-city ellipses). Kept as separate new columns
        # rather than touching Std_Lat/Std_Long (which stay exactly as
        # before, ddof=0 via .std()) so no already-reported number
        # changes -- this only adds the cross term needed to reconstruct
        # the TRUE (possibly rotated) ellipse downstream, e.g. in
        # terrain.screen_ellipse(), instead of the axis-aligned
        # mean+/-std approximation that ignored any correlation between a
        # city's longitude and latitude errors across bootstrap runs.
        if len(pts) >= 2:
            cov = np.cov(pts, rowvar=False)
            row['Var_Long'] = cov[0, 0]
            row['Var_Lat'] = cov[1, 1]
            row['Cov_LonLat'] = cov[0, 1]
        if actual_coords_by_name is not None:
            actual = actual_coords_by_name[name]
            row['Error_KM'] = train.calculate_distance_km(actual[0], actual[1], mean_c[0], mean_c[1])
        for col_name, coords_by_name in comparisons.items():
            comp = coords_by_name[name]
            row[col_name] = train.calculate_distance_km(mean_c[0], mean_c[1], comp[0], comp[1])
        rows.append(row)
    return pd.DataFrame(rows).sort_values('City').reset_index(drop=True)


def save_results(df, name):
    path = os.path.join(config.RESULTS_DIR, f"{name}.parquet")
    df.to_parquet(path, index=False)
    return path


def bootstrap_ensemble_mean_distribution(results, names, actual_coords_by_name,
                                          n_resamples=2000, master_seed=config.MASTER_SEED):
    """Nonparametric bootstrap over the already-trained replicate
    predictions, approximating the sampling distribution of the
    AGGREGATED predictor's overall error -- the exact quantity
    summarize_bootstrap() reports as Error_KM / "Overall Mean Error"
    (distance from the MEAN of the B replicate predictions to the truth,
    then averaged over cities), not the mean of B individual replicates'
    own errors -- those are not the same number and can differ
    substantially, since averaging predictions before scoring them
    cancels out some of each replicate's independent noise in a way that
    averaging their scores afterward does not.

    Why this test, and not a test of single-replicate performance: the
    paper's reported result IS the aggregated/bagged prediction (mean
    over B independently Poisson-resampled and retrained replicates --
    bagging, Breiman 1996), so the test needs to target the sampling
    variability of THAT statistic, not of a single untrained-together
    replicate. The standard nonparametric bootstrap for the sampling
    distribution of a statistic (here, "the error of a size-B average of
    replicate predictions") is to resample, with replacement, from the
    empirical distribution of those replicates -- treating the B
    predictions already on hand as the empirical population and drawing
    new size-B samples from it repeatedly (Efron, 1979, "Bootstrap
    Methods: Another Look at the Jackknife," Annals of Statistics 7(1);
    Efron & Tibshirani, 1993, An Introduction to the Bootstrap, chs. 6
    and 13, percentile confidence intervals for an arbitrary statistic
    computed from resampled data). No retraining is needed: the B
    replicate predictions per city, already produced by
    run_ancient_loo(), are exactly the empirical sample this resamples.

    All 15 cities' predictions for a given replicate b came from the
    SAME Poisson-perturbed matrix and the SAME training run (see
    run_loo_experiment()'s docstring: the outer loop is over bootstrap
    seeds, one full retraining per seed, all cities predicted together
    within it) -- so a replicate that happens to train unusually well or
    badly can move every city's error in the same direction at once.
    The resampling below draws ONE shared set of replicate indices per
    resample iteration and reuses it across every city, preserving that
    correlation, rather than resampling each city independently (which
    would treat different cities' errors as independent even when they
    came from the exact same retraining run, understating the resulting
    interval's width).

    Thin wrapper around bootstrap_per_city_errors() below -- that
    function does the actual resampling (single source of truth for it);
    this one just averages its per-city output across cities, which is
    all the one-sample test (significance_test_vs_reference()) needs."""
    return bootstrap_per_city_errors(
        results, names, actual_coords_by_name, n_resamples, master_seed,
    ).mean(axis=1)


def bootstrap_per_city_errors(results, names, actual_coords_by_name,
                               n_resamples=2000, master_seed=config.MASTER_SEED):
    """Identical resampling to bootstrap_ensemble_mean_distribution()
    above (one shared resample_idx per iteration, reused across cities,
    preserving cross-city correlation -- see that function's docstring),
    but returns the full (n_resamples, n_cities) error matrix instead of
    already averaging across cities. paired_significance_test_vs_
    reference() below needs each city's OWN resampled error to subtract
    that SAME city's own reference error before averaging across cities
    -- averaging across cities first (as the wrapper above does) would
    destroy the city-to-city pairing."""
    rng = np.random.RandomState(master_seed)
    n_cities = len(names)
    num_replicates = len(results[names[0]])
    resample_idx = rng.randint(0, num_replicates, size=(n_resamples, num_replicates))
    per_city_errors = np.zeros((n_resamples, n_cities))
    for j, name in enumerate(names):
        preds = np.asarray(results[name])  # (B, 2): [:, 0]=lon, [:, 1]=lat
        resampled_means = preds[resample_idx].mean(axis=1)  # (n_resamples, 2)
        actual_long, actual_lat = actual_coords_by_name[name]
        per_city_errors[:, j] = train.calculate_distance_km(
            resampled_means[:, 0], resampled_means[:, 1], actual_long, actual_lat,
        )
    return per_city_errors


def significance_test_vs_reference(resampled_errors, reference_km, reference_label):
    """One-sided nonparametric bootstrap significance test (percentile
    method -- Efron & Tibshirani, 1993, ch. 13): what fraction of
    `resampled_errors` (e.g. from bootstrap_ensemble_mean_distribution()
    above) are at or above `reference_km`? A small fraction is evidence
    the reported result is not a fluke of how the B replicates happened
    to average out, but holds up under resampling.

    This is a one-sample test against a single fixed external number
    (config.BARJAMOVIC_LOO_HEADLINE_KM). An earlier version of this
    docstring claimed this was the only test possible because Barjamovic
    et al. publish their out-of-sample error as one aggregate figure with
    no city-by-city breakdown -- that premise was wrong. Their own
    per-city known-city estimates ARE available (data.
    BARJAMOVIC_KNOWN_ESTIMATES, their Appendix Table 2 robustness
    exercise -- verified to reproduce their own printed per-city errors,
    e.g. Hattus ~=133.3 km, to within rounding, confirming 116.03 really
    is the mean of a real per-city table). See
    paired_significance_test_vs_reference() below for the city-matched
    version this makes possible, which is kept as an ADDITIONAL result
    alongside this one, not a replacement -- the two test different
    things (aggregate-vs-aggregate here; matched, city-by-city there) and
    both are informative."""
    ci_low, ci_high = np.percentile(resampled_errors, [2.5, 97.5])
    return {
        'reference_label': reference_label,
        'reference_km': reference_km,
        'bootstrap_mean_km': float(np.mean(resampled_errors)),
        'bootstrap_std_km': float(np.std(resampled_errors)),
        'ci_95_low_km': float(ci_low),
        'ci_95_high_km': float(ci_high),
        'n_resamples': len(resampled_errors),
        'p_value_one_sided': float(np.mean(resampled_errors >= reference_km)),
    }


def save_significance(sig, name):
    path = os.path.join(config.RESULTS_DIR, f"{name}_significance.csv")
    pd.Series(sig).to_frame('value').to_csv(path)
    return path


def barjamovic_known_city_errors():
    """Barjamovic et al.'s own per-city error for each of the 15 known
    cities: distance between their own robustness-exercise estimate
    (data.barjamovic_known_lookup(), their Appendix Table 2) and that
    city's true coordinate. This is the per-city breakdown behind their
    published 116.03 km mean (config.BARJAMOVIC_LOO_HEADLINE_KM) --
    spot-verified to reproduce their own printed per-city figures (e.g.
    Hattus ~=133.3 km) to within rounding, confirming the mean really is
    the average of a genuine per-city table, not a number published
    without one. Used by paired_significance_test_vs_reference() below."""
    est_lookup = data.barjamovic_known_lookup()
    known_coords, known_names = data.ancient_known_coords()
    errors = {}
    for i, name in enumerate(known_names):
        if name not in est_lookup:
            continue
        est_long, est_lat = est_lookup[name]
        true_long, true_lat = known_coords[i]
        errors[name] = float(train.calculate_distance_km(est_long, est_lat, true_long, true_lat))
    return errors


def paired_significance_test_vs_reference(point_estimate_errors_by_city, reference_errors_by_city, names):
    """Paired comparison against Barjamovic et al.'s own PER-CITY known-
    city estimates (barjamovic_known_city_errors() above), instead of
    only their single published aggregate reference figure -- see
    significance_test_vs_reference()'s docstring for why this is possible.

    Classical Wilcoxon signed-rank test (scipy.stats.wilcoxon, one-sided,
    EXACT method) on the point estimates: this method's own per-city
    error (the distance from the bootstrap-mean prediction -- the
    "middle point" of the B replicates -- to the true coordinate; exactly
    what summarize_bootstrap() reports as Error_KM per city) vs.
    Barjamovic's same per-city error, matched city for city. This is the
    standard test for comparing two methods' errors across the same N
    paired test cases -- see Demsar (2006), "Statistical Comparisons of
    Classifiers over Multiple Data Sets," Journal of Machine Learning
    Research 7, 1-30, the standard reference for using the Wilcoxon
    signed-rank test to compare two methods across paired benchmark
    items, which is exactly this situation (15 known cities as the
    matched items).

    method='exact' is passed explicitly rather than left to scipy's
    default ('auto', which falls back to a large-sample normal
    approximation once n or the presence of ties/zero-differences makes
    the exact permutation distribution impractical) -- at n=15 with
    continuous-valued km errors (no ties, no exact-zero differences
    expected), the exact distribution is both computationally feasible
    and the statistically correct choice.

    n=15 known cities is a modest sample -- this affects the test's
    POWER (its ability to detect a real but small effect), not its
    validity. The result is a paired comparison specific to this exact,
    complete set of 15 known Bronze Age cities with recoverable
    coordinates, not a claim about how the two methods would compare on
    some different or larger set of cities.

    point_estimate_errors_by_city: {city_name: float} km error per city
        for THIS method, e.g. dict(zip(summary_df['City'],
        summary_df['Error_KM'])).
    reference_errors_by_city: {city_name: float} km error per city for
        Barjamovic et al., e.g. from barjamovic_known_city_errors()
        above.
    names: city names to include, in any order (both dicts are indexed
        by name, not position)."""
    point_estimates = np.array([point_estimate_errors_by_city[name] for name in names])
    reference_vec = np.array([reference_errors_by_city[name] for name in names])
    try:
        # scipy >=1.9 renamed the exact/asymptotic-method switch from
        # `mode` to `method`; try the current name first.
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(
            point_estimates, reference_vec, alternative='less', method='exact',
        )
    except TypeError:
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(
            point_estimates, reference_vec, alternative='less', mode='exact',
        )

    return {
        'n_cities_paired': len(names),
        'mean_paired_diff_km': float(np.mean(point_estimates - reference_vec)),
        'wilcoxon_statistic': float(wilcoxon_stat),
        'wilcoxon_p_value_one_sided': float(wilcoxon_p),
    }


def run_ancient_loo(metric, hparams, n_bootstrap, master_seed=config.MASTER_SEED, weight_decay=5e-4):
    """15 fictitiously-held-out known cities, LOO, on the full 25-node
    ancient graph (paper Table 4/9 -- Dyadic; TII counterpart)."""
    known_idx, lost_idx, known_names, lost_names = data.ancient_known_lost_split()
    known_coords, ordered_known_names = data.ancient_known_coords()
    num_nodes = len(data.ANCIENT_CITY_NAMES)

    # Both metrics now consume the same RAW (directed) matrix -- dyadic_weights()
    # and tii_weights_from_directed() both self-symmetrize from directional
    # input (see weights.py); no presymmetrization needed for either.
    base_matrix = data.ANCIENT_MATRIX_DATA
    if metric == 'dyadic':
        weight_fn = weights.dyadic_weights
    elif metric == 'tii':
        weight_fn = weights.tii_weights_from_directed
    else:
        raise ValueError(f"unknown metric: {metric}")

    def resample_fn(seed):
        return train.resample_poisson(base_matrix, seed)

    results, scaler = run_loo_experiment(
        num_nodes, resample_fn, weight_fn, known_idx, known_coords, ordered_known_names,
        hparams, n_bootstrap, master_seed, desc=f"Ancient LOO ({metric})", weight_decay=weight_decay,
    )
    actual = {name: known_coords[i] for i, name in enumerate(ordered_known_names)}
    summary_df = summarize_bootstrap(results, ordered_known_names, actual_coords_by_name=actual)
    return summary_df, results


def run_modern_loo(metric, hparams, n_bootstrap, master_seed=config.MASTER_SEED, poisson_cap=1e3):
    """15 modern Turkish province centers, LOO (no lost cities exist here,
    so Bug 2 does not apply -- the graph is already exactly the known
    set)."""
    valid_cities, known_coords = data.modern_cities_and_coords()
    num_nodes = len(valid_cities)

    # dyadic_weights() self-symmetrizes from a RAW (directed) matrix (see
    # weights.py); tii_weights_from_symmetric() requires an already-
    # presymmetrized input (it does not self-symmetrize). Different base
    # matrices for the two metrics as a result.
    if metric == 'dyadic':
        base_matrix = data.load_modern_matrix(valid_cities=valid_cities)
        weight_fn = weights.dyadic_weights
    elif metric == 'tii':
        base_matrix = data.load_modern_symmetric_matrix(valid_cities=valid_cities)
        weight_fn = weights.tii_weights_from_symmetric
    else:
        raise ValueError(f"unknown metric: {metric}")

    max_val = base_matrix.max()
    scaling_factor = poisson_cap / max_val if max_val > poisson_cap else 1.0
    scaled_base_matrix = base_matrix * scaling_factor

    known_idx = np.arange(num_nodes)

    def resample_fn(seed):
        return train.resample_poisson(scaled_base_matrix, seed)

    results, scaler = run_loo_experiment(
        num_nodes, resample_fn, weight_fn, known_idx, known_coords, valid_cities,
        hparams, n_bootstrap, master_seed, desc=f"Modern LOO ({metric})",
    )
    actual = {name: known_coords[i] for i, name in enumerate(valid_cities)}
    summary_df = summarize_bootstrap(results, valid_cities, actual_coords_by_name=actual)
    return summary_df, results


def run_ancient_lost_prediction(hparams, n_bootstrap, master_seed=config.MASTER_SEED,
                                 constraint_fn=None, gamma=0.0, weight_decay=5e-4):
    """Final prediction for the 10 actually-lost cities on the full 25-node
    graph, supervised by all 15 known cities every run (paper Table 11/12).
    Bug 1 fix: uses the same dyadic_weights() and the same resampling base
    as run_ancient_loo('dyadic'), instead of the old import-only formula.

    constraint_fn/gamma: optional §7.1 directional-constraint penalty (see
    src/constraints.py). Default (None, 0.0) is the plain unconstrained
    run this function has always produced -- see
    experiments/ancient_lost_dyadic_constrained.py for the constrained
    variant, which is a separate, additional result, not a replacement."""
    known_idx, lost_idx, known_names, lost_names = data.ancient_known_lost_split()
    known_coords, ordered_known_names = data.ancient_known_coords()
    num_nodes = len(data.ANCIENT_CITY_NAMES)
    base_matrix = data.ANCIENT_MATRIX_DATA

    rng = np.random.RandomState(master_seed)
    bootstrap_seeds = rng.randint(0, 10000, size=n_bootstrap)

    scaler = MinMaxScaler().fit(known_coords)
    target_scaled = torch.tensor(scaler.transform(known_coords), dtype=torch.float)
    known_idx_arr = np.asarray(known_idx)
    train_mask_local = np.arange(len(known_idx_arr))
    lost_idx_arr = np.asarray(lost_idx)

    results = {name: [] for name in lost_names}
    for seed in tqdm(bootstrap_seeds, desc="Lost-city prediction (dyadic)"):
        resampled = train.resample_poisson(base_matrix, seed)
        weight_matrix = weights.dyadic_weights(resampled)
        graph_data = data.create_graph_data(weight_matrix, num_nodes)

        preds = train.fit_predict(
            graph_data, num_nodes, hparams['embedding_dim'], hparams['epochs'],
            hparams['lr'], seed, target_scaled, scaler,
            known_idx_arr, train_mask_local, lost_idx_arr,
            constraint_fn=constraint_fn, gamma=gamma, weight_decay=weight_decay,
        )
        for i, name in enumerate(lost_names):
            results[name].append(preds[i])

    comparisons = {
        'Dist_to_Baj': data.barjamovic_lost_lookup(),          # Barjamovic et al.'s gravity model (QJE paper)
        'Dist_to_Barjamovic2011': data.barjamovic_2011_lookup(),  # Barjamovic's own 2011 historical monograph
        'Dist_to_Forlanini2008': data.forlanini_2008_lookup(),
    }
    summary_df = summarize_bootstrap(results, lost_names, comparisons=comparisons)
    return summary_df, results


# =============================================================================
# MDS drivers (paper Sec. 2.1: SMACOF embedding + weighted Procrustes,
# src/mds.py, instead of GCN's message-passing). Mirror the three GCN
# drivers above exactly in control flow -- outer loop over bootstrap seeds,
# inner loop over LOO folds / lost cities -- swapping train.fit_predict for
# one shared mds.smacof_embed() per replicate plus a per-target
# mds.weighted_procrustes(). summarize_bootstrap(), save_results(),
# bootstrap_ensemble_mean_distribution(), significance_test_vs_reference()
# above are method-agnostic (operate on {name: [[lon,lat],...]} result
# dicts) and are reused completely unchanged -- MDS gets the same
# significance test as GCN for free.
#
# Bug M4 fix (found during this refactor, not in the original brief's list)
# ---------------------------------------------------------------------------
# The legacy MDSDyadicBootstrap.py / MDSTIIBootstrap.py (ancient LOO
# training) built their similarity matrix from a `cities_df` listing only
# the 15 KNOWN cities, then filtered `if c1 in all_cities and c2 in
# all_cities` -- silently dropping every trade record touching one of the
# 10 lost cities, i.e. exactly GCN's own original Bug 2 (see this file's
# header docstring), just in the opposite script: MDS's TRAINING scripts
# had it, while MDS's final lost-city script (MDSDyadicResults.py) already
# correctly used the full 25-city matrix_data. run_loo_experiment_mds()
# below always embeds the full `all_names` set (data.ANCIENT_CITY_NAMES,
# 25 entries for the ancient network) and only restricts which names get
# LOO-supervised, never which cities exist in the embedding -- same fix,
# same rationale as GCN's Bug 2: a lost city's trade edges are real
# topological signal for its known neighbors' placement.
# =============================================================================

def run_loo_experiment_mds(all_names, resample_fn, weight_fn, known_names, known_coords,
                            n_bootstrap, master_seed=config.MASTER_SEED,
                            desc="MDS Monte Carlo Iterations"):
    """Generic MDS leave-one-out bootstrap. `all_names` is the full ordered
    city-name list the similarity/dissimilarity matrix and SMACOF embedding
    are built over (Bug M4: never a known-cities-only subset); `known_names`
    (a subset of all_names, row-aligned with known_coords) is LOO'd one at a
    time, each fold's anchor set being every other name in known_names."""
    rng = np.random.RandomState(master_seed)
    bootstrap_seeds = rng.randint(0, 10000, size=n_bootstrap)

    name_to_pos = {name: i for i, name in enumerate(all_names)}
    coord_by_name = {name: known_coords[i] for i, name in enumerate(known_names)}
    results = {name: [] for name in known_names}

    for seed in tqdm(bootstrap_seeds, desc=desc):
        raw_matrix = resample_fn(seed)
        weight_matrix = weight_fn(raw_matrix)
        dissim = mds.dissimilarity_from_weights(weight_matrix)
        embedding = mds.smacof_embed(dissim, random_state=seed)

        for target_name in known_names:
            anchor_names = [n for n in known_names if n != target_name]
            anchor_positions = [name_to_pos[n] for n in anchor_names]
            target_pos = name_to_pos[target_name]

            anchor_real = np.array([coord_by_name[n] for n in anchor_names])
            anchor_mds = embedding[anchor_positions]
            raw_w = weight_matrix[target_pos, anchor_positions]

            pred = mds.weighted_procrustes(anchor_real, anchor_mds, embedding[target_pos], raw_w)
            results[target_name].append(pred)

    return results


def run_ancient_loo_mds(metric, n_bootstrap, master_seed=config.MASTER_SEED):
    """15 fictitiously-held-out known cities, LOO, on the full 25-city
    ancient similarity matrix (paper Table 6/7 -- Dyadic/TII). MDS analog of
    run_ancient_loo() above; see this module's Bug M4 note for why the
    embedding always uses all 25 names."""
    known_idx, lost_idx, known_names, lost_names = data.ancient_known_lost_split()
    known_coords, ordered_known_names = data.ancient_known_coords()
    all_names = data.ANCIENT_CITY_NAMES

    # Both metrics now consume the same RAW (directed) matrix -- dyadic_weights()
    # and symmetric_tii_from_directed() both self-symmetrize from directional
    # input (see weights.py), so plain per-cell resampling is safe for both
    # (no more need for mds.resample_poisson_symmetric() here -- that was
    # only needed by the old presymmetrize-first dyadic formula).
    base_matrix = data.ANCIENT_MATRIX_DATA
    if metric == 'dyadic':
        weight_fn = weights.dyadic_weights
    elif metric == 'tii':
        weight_fn = weights.symmetric_tii_from_directed
    else:
        raise ValueError(f"unknown metric: {metric}")

    def resample_fn(seed):
        return train.resample_poisson(base_matrix, seed)

    results = run_loo_experiment_mds(
        all_names, resample_fn, weight_fn, ordered_known_names, known_coords,
        n_bootstrap, master_seed, desc=f"Ancient LOO MDS ({metric})",
    )
    actual = {name: known_coords[i] for i, name in enumerate(ordered_known_names)}
    summary_df = summarize_bootstrap(results, ordered_known_names, actual_coords_by_name=actual)
    return summary_df, results


def run_modern_loo_mds(metric, n_bootstrap, master_seed=config.MASTER_SEED, poisson_cap=1e3):
    """15 modern Turkish province centers, LOO (no lost cities exist here,
    so Bug M4 does not apply -- the embedded set is already exactly the
    known set). MDS analog of run_modern_loo() above."""
    valid_cities, known_coords = data.modern_cities_and_coords()

    # dyadic_weights() self-symmetrizes from a RAW (directed) matrix (see
    # weights.py); symmetric_tii_from_symmetric() requires an already-
    # presymmetrized input and does NOT self-symmetrize -- still needs
    # mds.resample_poisson_symmetric() (Bug M6) to guarantee the exact
    # symmetry sklearn's MDS requires.
    if metric == 'dyadic':
        base_matrix = data.load_modern_matrix(valid_cities=valid_cities)
        weight_fn = weights.dyadic_weights
        needs_paired_resample = False
    elif metric == 'tii':
        base_matrix = data.load_modern_symmetric_matrix(valid_cities=valid_cities)
        weight_fn = weights.symmetric_tii_from_symmetric
        needs_paired_resample = True
    else:
        raise ValueError(f"unknown metric: {metric}")

    max_val = base_matrix.max()
    scaling_factor = poisson_cap / max_val if max_val > poisson_cap else 1.0
    scaled_base_matrix = base_matrix * scaling_factor

    if needs_paired_resample:
        def resample_fn(seed):
            return mds.resample_poisson_symmetric(scaled_base_matrix, seed)
    else:
        def resample_fn(seed):
            return train.resample_poisson(scaled_base_matrix, seed)

    results = run_loo_experiment_mds(
        valid_cities, resample_fn, weight_fn, valid_cities, known_coords,
        n_bootstrap, master_seed, desc=f"Modern LOO MDS ({metric})",
    )
    actual = {name: known_coords[i] for i, name in enumerate(valid_cities)}
    summary_df = summarize_bootstrap(results, valid_cities, actual_coords_by_name=actual)
    return summary_df, results


def run_ancient_lost_prediction_mds(n_bootstrap, master_seed=config.MASTER_SEED):
    """Final prediction for the 10 actually-lost cities on the full 25-city
    ancient network, aligned every replicate using all 15 known cities as
    anchors (paper Table 8). MDS analog of run_ancient_lost_prediction()
    above.

    Bug M1 fix: uses weights.dyadic_weights() (paper Eq. 2-3) instead of the
    legacy script's import-share-only formula.
    Bug M2 fix: uses mds.weighted_procrustes() with the same
    trade-derived weight vector as every LOO script, instead of the legacy
    script's unweighted alignment.
    Bug M3 fix: resampling goes through train.resample_poisson(seed),
    independently seeded per replicate, instead of an unseeded/desynced
    np.random.poisson() call.
    """
    known_idx, lost_idx, known_names, lost_names = data.ancient_known_lost_split()
    known_coords, ordered_known_names = data.ancient_known_coords()
    all_names = data.ANCIENT_CITY_NAMES
    base_matrix = data.ANCIENT_MATRIX_DATA

    rng = np.random.RandomState(master_seed)
    bootstrap_seeds = rng.randint(0, 10000, size=n_bootstrap)

    name_to_pos = {name: i for i, name in enumerate(all_names)}
    coord_by_name = {name: known_coords[i] for i, name in enumerate(ordered_known_names)}
    anchor_positions = [name_to_pos[n] for n in ordered_known_names]
    anchor_real = np.array([coord_by_name[n] for n in ordered_known_names])

    results = {name: [] for name in lost_names}
    for seed in tqdm(bootstrap_seeds, desc="Lost-city prediction (MDS dyadic)"):
        # dyadic_weights() self-symmetrizes from the raw directed matrix --
        # plain resampling is safe (see weights.py).
        raw_matrix = train.resample_poisson(base_matrix, seed)
        weight_matrix = weights.dyadic_weights(raw_matrix)
        dissim = mds.dissimilarity_from_weights(weight_matrix)
        embedding = mds.smacof_embed(dissim, random_state=seed)
        anchor_mds = embedding[anchor_positions]

        for name in lost_names:
            target_pos = name_to_pos[name]
            raw_w = weight_matrix[target_pos, anchor_positions]
            pred = mds.weighted_procrustes(anchor_real, anchor_mds, embedding[target_pos], raw_w)
            results[name].append(pred)

    comparisons = {
        'Dist_to_Baj': data.barjamovic_lost_lookup(),
        'Dist_to_Barjamovic2011': data.barjamovic_2011_lookup(),
        'Dist_to_Forlanini2008': data.forlanini_2008_lookup(),
    }
    summary_df = summarize_bootstrap(results, lost_names, comparisons=comparisons)
    return summary_df, results
