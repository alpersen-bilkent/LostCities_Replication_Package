# -*- coding: utf-8 -*-
"""
Synthetic verification for src/mds.py and the MDS-specific parts of
src/weights.py, run BEFORE trusting any real experiment rerun (per the
project's standing rule: show a fix is right, don't assume it from reading
the diff). Each test either:
  (a) proves the new src/ code reproduces a legacy script's own formula
      exactly, on a small hand-checkable synthetic example (refactor
      fidelity), or
  (b) proves a claimed bug (M1/M2/M3) actually changes the numeric result,
      not just cosmetically.

Run directly: `python tests/test_mds.py` (no pytest dependency -- plain
asserts, so it works even in a minimal environment). Every assert prints
its own PASS line so a partial run still shows progress.

Run via C:\ProgramData\anaconda3\python.exe (the install Spyder actually
uses on this machine) -- a different, separate anaconda3 install under
this Windows user's own profile is broken (missing its own Lib/encodings);
not relevant once you're pointed at the right one.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src import mds, weights
from src.train import resample_poisson


def _ok(name):
    print(f"PASS: {name}")


# ---------------------------------------------------------------------------
# 1. Dyadic share: does weights.dyadic_weights() implement Barbieri's own
#    published formula (Barbieri 1996; Gartzke & Li 2003, Table I -- the
#    paper's own citations [21]/[22])? trade_share_i = (imports_ij +
#    exports_ij) / (imports_i + exports_i): the numerator is the TOTAL
#    bilateral volume between i and j (both directions summed), used
#    IDENTICALLY in both i's and j's share term -- only the denominator
#    (each city's own total trade) differs between the two terms. This was
#    briefly changed (same day, MDS session) to a directional numerator
#    (X_ij/V_i + X_ji/V_j, each direction's own raw count) on the theory
#    that this matches the paper's own Eq. 3 as literally typeset -- checked
#    directly against the primary source and reverted; see weights.py's
#    module docstring "Dyadic-formula correction, reverted" for the full
#    citation trail. This test asserts the CURRENT (total-bilateral-volume,
#    reverted) formula -- if it's asserting the directional one again,
#    someone re-broke this.
# ---------------------------------------------------------------------------

def test_dyadic_share_matches_barbieri_total_bilateral_volume():
    # A tiny 4-city network with GENUINELY DIRECTED trade (X_ij != X_ji for
    # most pairs) -- matching the paper's own description of the raw data
    # ("since the raw trade data is directional, X_ij != X_ji"). Chosen so
    # the total-bilateral-volume and directional-only formulas give
    # numerically different answers, so this test can't pass by accident.
    n = 4
    raw = np.array([
        [0, 5, 0, 2],
        [3, 0, 3, 0],
        [0, 1, 0, 7],
        [4, 0, 2, 0],
    ], dtype=float)
    assert not np.allclose(raw, raw.T), "test fixture must be genuinely directed"

    # --- Barbieri 1996 / Gartzke & Li 2003, transcribed from the primary
    #     source: vol_ij = X_ij + X_ji (total bilateral volume, both
    #     directions), used identically in both i's and j's term ---
    V = raw.sum(axis=1) + raw.sum(axis=0)  # total volume per city, both directions
    expected_share = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            vol_ij = raw[i, j] + raw[j, i]
            if vol_ij > 0:
                expected_share[i, j] = 0.5 * (vol_ij / V[i] + vol_ij / V[j])

    # --- new src/ code ---
    new_share = weights.dyadic_weights(raw)

    assert np.allclose(expected_share, new_share), (expected_share, new_share)
    _ok("dyadic_weights() matches Barbieri's total-bilateral-volume formula")

    assert np.allclose(new_share, new_share.T), (
        "dyadic_weights() must produce a symmetric output even from directed "
        "input -- both terms share the same (symmetric) numerator by "
        "construction, so this should hold trivially."
    )
    _ok("confirmed dyadic_weights() output is symmetric despite directed input")

    # And confirm this is genuinely NOT the directional-numerator formula
    # that was reverted -- guards against silently re-introducing that bug.
    directional_share = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if raw[i, j] > 0 or raw[j, i] > 0:
                directional_share[i, j] = 0.5 * (raw[i, j] / V[i] + raw[j, i] / V[j])
    assert not np.allclose(new_share, directional_share), (
        "dyadic_weights() matches the directional-numerator formula that was "
        "reverted -- the total-bilateral-volume fix may have been undone."
    )
    _ok("confirmed dyadic_weights() is NOT the reverted directional-numerator formula")

    # And the dissimilarity conversion (paper Eq. 4): delta = sqrt(1/(I+eps))
    new_dissim = mds.dissimilarity_from_weights(new_share)
    mask = new_share > 0
    expected_dissim = np.sqrt(1.0 / (new_share[mask] + 1e-9))
    assert np.allclose(new_dissim[mask], expected_dissim, atol=1e-4), (
        new_dissim, expected_dissim,
    )
    assert np.allclose(new_dissim, new_dissim.T), "dissimilarity matrix must be symmetric"
    _ok("dissimilarity_from_weights() matches paper Eq.4 and stays symmetric")


# ---------------------------------------------------------------------------
# 2. TII: does weights.symmetric_tii_from_directed() (the RAW, pre-log-norm
#    score) match MDSTIIBootstrap.py's own T_ij/T_ji formula? Crucially:
#    MDS's own TII dissimilarity (paper Eq. 5-7) has NO log/min-max step --
#    that transform is GCN-specific (Eq. 9, Sec. 2.2.1). Using
#    weights.tii_weights_from_directed() (the GCN-ready, log-normalized
#    version) for MDS would silently NOT match the paper's MDS methodology.
#    This test guards against exactly that mistake.
# ---------------------------------------------------------------------------

def test_tii_share_matches_legacy_ancient_loo_and_has_no_lognorm():
    n = 4
    rng = np.random.RandomState(0)
    d_mat = rng.randint(0, 10, size=(n, n)).astype(float)
    np.fill_diagonal(d_mat, 0)

    # --- legacy MDSTIIBootstrap.py's own formula, transcribed verbatim ---
    total_exports = d_mat.sum(axis=1)
    total_imports = d_mat.sum(axis=0)
    world_trade = d_mat.sum()
    T_ij = (d_mat * world_trade) / (np.outer(total_exports, total_imports) + 1e-9)
    T_ji = (d_mat.T * world_trade) / (np.outer(total_imports, total_exports) + 1e-9)
    legacy_sim = 0.5 * (T_ij + T_ji)

    # --- new src/ code: the RAW (pre-lognorm) symmetric TII ---
    new_sim = weights.symmetric_tii_from_directed(d_mat)

    assert np.allclose(legacy_sim, new_sim), (legacy_sim, new_sim)
    _ok("symmetric_tii_from_directed() matches legacy ancient-LOO TII formula")

    # Sanity check the "no log-normalization" claim itself: the GCN-ready
    # tii_weights_from_directed() must NOT equal the raw score (it applies
    # log1p + min-max on top), confirming these are genuinely different
    # functions and MDS must use the raw one.
    gcn_ready = weights.tii_weights_from_directed(d_mat)
    assert not np.allclose(gcn_ready, new_sim), (
        "tii_weights_from_directed() should differ from the raw symmetric "
        "TII score (it applies GCN's Eq.9 log+min-max transform) -- if "
        "these match, someone made _lognorm_edges() a no-op and MDS's TII "
        "dissimilarity would be silently wrong if pointed at this function."
    )
    _ok("confirmed tii_weights_from_directed() != raw TII (lognorm is doing something)")


# ---------------------------------------------------------------------------
# 3. Weighted Procrustes: does mds.weighted_procrustes() reproduce the
#    legacy scripts' own per-target alignment step exactly?
# ---------------------------------------------------------------------------

def test_weighted_procrustes_matches_legacy_loop():
    rng = np.random.RandomState(1)
    n_anchor = 6
    known_real = rng.uniform(30, 42, size=(n_anchor, 2))
    known_mds = rng.uniform(-5, 5, size=(n_anchor, 2))
    target_mds_point = rng.uniform(-5, 5, size=2)
    raw_w = rng.uniform(0.1, 20.0, size=n_anchor)

    # --- legacy scripts' own inline loop body, transcribed verbatim ---
    w_vec = (raw_w - raw_w.min()) / (raw_w.max() - raw_w.min() + 1e-9) + 0.1
    W = np.diag(w_vec)
    c1 = np.average(known_real, axis=0, weights=w_vec)
    c2 = np.average(known_mds, axis=0, weights=w_vec)
    A = (known_real - c1).T @ W @ (known_mds - c2)
    U, _, Vt = np.linalg.svd(A)
    R = Vt.T @ U.T
    scale = np.linalg.norm(known_real - c1) / np.linalg.norm(known_mds - c2)
    legacy_est = (target_mds_point - c2) @ R.T * scale + c1

    # --- new src/ code ---
    new_est = mds.weighted_procrustes(known_real, known_mds, target_mds_point, raw_w)

    assert np.allclose(legacy_est, new_est), (legacy_est, new_est)
    _ok("weighted_procrustes() matches legacy per-target alignment loop")


# ---------------------------------------------------------------------------
# 4. Bug M1: prove MDSDyadicResults.py's column-import-share formula is not
#    just cosmetically different from the paper-correct dyadic_weights(),
#    but actually changes the resulting similarity matrix numerically, on a
#    genuinely asymmetric (directed) trade matrix like the ancient one.
# ---------------------------------------------------------------------------

def test_bug_M1_final_prediction_formula_actually_diverges():
    n = 5
    rng = np.random.RandomState(2)
    b_matrix = rng.randint(0, 10, size=(n, n)).astype(float)
    np.fill_diagonal(b_matrix, 0)
    b_matrix[b_matrix == 0] = 0.5  # legacy script's own zero-fill

    # --- legacy MDSDyadicResults.py's own (buggy) formula ---
    col_sums = b_matrix.sum(axis=0)
    S_matrix = b_matrix / col_sums
    legacy_I = 0.5 * (S_matrix + S_matrix.T)

    # --- correct formula (Barbieri 1996 total-bilateral-volume, Eq. 2-3), on
    #     the raw directed matrix ---
    correct_I = weights.dyadic_weights(b_matrix)

    assert not np.allclose(legacy_I, correct_I), (
        "Bug M1 claim requires these to differ on this fixture -- if they "
        "match, the bug characterization above needs revisiting before "
        "trusting the fix."
    )
    _ok("confirmed Bug M1: legacy final-prediction formula diverges from Eq.2-3")


# ---------------------------------------------------------------------------
# 5. Bug M2: prove weighted vs. unweighted Procrustes give materially
#    different predictions when anchor weights are non-uniform (i.e. that
#    MDSDyadicResults.py dropping the weight matrix entirely was not a
#    harmless simplification).
# ---------------------------------------------------------------------------

def test_bug_M2_unweighted_procrustes_diverges_from_weighted():
    rng = np.random.RandomState(3)
    n_anchor = 8
    known_real = rng.uniform(30, 42, size=(n_anchor, 2))
    known_mds = rng.uniform(-5, 5, size=(n_anchor, 2))
    target_mds_point = rng.uniform(-5, 5, size=2)
    # deliberately skewed weights (one anchor dominates), like a real trade
    # network where one neighbor trades far more with the target than others
    raw_w = np.array([50.0, 1, 1, 1, 1, 1, 1, 1])

    weighted_est = mds.weighted_procrustes(known_real, known_mds, target_mds_point, raw_w)
    unweighted_est = mds.weighted_procrustes(known_real, known_mds, target_mds_point, None)

    diff_km_scale = np.linalg.norm(weighted_est - unweighted_est)
    assert diff_km_scale > 0.05, (
        "Bug M2 claim requires weighted and unweighted Procrustes to give "
        f"materially different predictions on a skewed-weight fixture; got "
        f"a difference of only {diff_km_scale} degrees, which would suggest "
        "the bug doesn't actually change results on real data."
    )
    _ok(f"confirmed Bug M2: weighted vs unweighted Procrustes diverge by {diff_km_scale:.3f} deg")


# ---------------------------------------------------------------------------
# 6. Bug M3: prove src/train.py:resample_poisson(seed) is independent of
#    ambient global NumPy random state, unlike the legacy scripts' bare
#    np.random.poisson(...) calls.
# ---------------------------------------------------------------------------

def test_bug_M3_resample_poisson_is_seed_independent_of_global_state():
    base_matrix = np.array([[0, 5, 2], [5, 0, 3], [2, 3, 0]], dtype=float)

    np.random.seed(123)
    fresh = resample_poisson(base_matrix, seed=99)

    np.random.seed(123)
    np.random.poisson(base_matrix)  # consume some of the global stream first
    np.random.rand(17)              # ...and some more, unrelated draws
    after_noise = resample_poisson(base_matrix, seed=99)

    assert np.array_equal(fresh, after_noise), (
        "resample_poisson(seed) must give an identical result regardless of "
        "prior global-RNG consumption -- if this fails, it's no longer "
        "independently seeded and Bug M3 is not actually fixed."
    )
    _ok("confirmed Bug M3 fix: resample_poisson(seed) is independent of global RNG state")

    # And demonstrate the legacy failure mode directly: two bare
    # np.random.poisson(base_matrix) calls under the same global seed(123)
    # give DIFFERENT results once *any* other draw happens in between --
    # this is exactly what happens across replicates in the unfixed scripts.
    np.random.seed(123)
    legacy_a = np.random.poisson(base_matrix)
    np.random.seed(123)
    np.random.rand(1)  # simulates "some other draw happened first"
    legacy_b = np.random.poisson(base_matrix)
    assert not np.array_equal(legacy_a, legacy_b), (
        "expected the legacy (unseeded-per-call) pattern to be sensitive to "
        "prior draws -- if this now passes as equal, the demonstration "
        "fixture itself needs revisiting."
    )
    _ok("confirmed legacy pattern IS sensitive to prior global-RNG draws (the bug is real)")


# ---------------------------------------------------------------------------
# 7. Bug M6 (found by actually running the real experiment scripts, not by
#    reading): mds.resample_poisson_symmetric() must produce an EXACTLY
#    symmetric matrix -- sklearn's MDS(dissimilarity='precomputed') raises
#    ValueError on any asymmetry, and train.resample_poisson() (independent
#    per-cell draws) cannot guarantee that on an already-symmetric base
#    matrix, since positions (i,j)/(j,i) get two independent Poisson draws
#    of the same lambda instead of one shared draw.
# ---------------------------------------------------------------------------

def test_bug_M6_resample_poisson_symmetric_is_exactly_symmetric():
    base = np.array([
        [0, 5, 2, 0],
        [5, 0, 3, 7],
        [2, 3, 0, 1],
        [0, 7, 1, 0],
    ], dtype=float)
    assert np.array_equal(base, base.T), "test fixture must be symmetric"

    resampled = mds.resample_poisson_symmetric(base, seed=42)
    assert np.array_equal(resampled, resampled.T), (
        "resample_poisson_symmetric() must be exactly symmetric -- sklearn's "
        "MDS(dissimilarity='precomputed') will otherwise raise ValueError."
    )
    _ok("confirmed Bug M6 fix: resample_poisson_symmetric() is exactly symmetric")

    # And confirm the failure mode it replaces is real: plain elementwise
    # resample_poisson() on the SAME symmetric base is generally NOT
    # symmetric (two independent draws per mirrored pair).
    from src.train import resample_poisson
    n_trials_with_asymmetry = 0
    for trial_seed in range(20):
        naive = resample_poisson(base, seed=trial_seed)
        if not np.array_equal(naive, naive.T):
            n_trials_with_asymmetry += 1
    assert n_trials_with_asymmetry > 0, (
        "expected plain resample_poisson() on a symmetric base to produce "
        "an asymmetric result at least sometimes across 20 seeds -- if this "
        "never happens, the Bug M6 characterization needs revisiting."
    )
    _ok(f"confirmed plain resample_poisson() IS asymmetric in "
        f"{n_trials_with_asymmetry}/20 trials (the bug is real)")


if __name__ == '__main__':
    test_dyadic_share_matches_barbieri_total_bilateral_volume()
    test_tii_share_matches_legacy_ancient_loo_and_has_no_lognorm()
    test_weighted_procrustes_matches_legacy_loop()
    test_bug_M1_final_prediction_formula_actually_diverges()
    test_bug_M2_unweighted_procrustes_diverges_from_weighted()
    test_bug_M3_resample_poisson_is_seed_independent_of_global_state()
    test_bug_M6_resample_poisson_symmetric_is_exactly_symmetric()
    print("\nAll MDS synthetic checks passed.")
