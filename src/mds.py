# -*- coding: utf-8 -*-
"""
MDS-specific pipeline: SMACOF embedding and weighted Procrustes alignment
(paper Sec. 2.1 -- Eq. 4/7 for the dissimilarity conversion, Sec. 2.1.2 for
the target-specific weighted Procrustes alignment). This is the ONLY place
either step is defined, so "did the LOO scripts and the final lost-city
prediction use the same alignment method" is answerable by reading this one
file, the same way src/weights.py answers it for edge weights.

Bug M1 fix
----------
The original MDSDyadicResults.py (final, headline lost-city prediction)
built its similarity matrix as S = matrix / col_sums (share of the
*destination city's total imports only*) then symmetrized -- not the
paper's own Eq. 2-3 (I_ij = 0.5*(X_ij/V_i + X_ji/V_j), V_i = symmetrized
total volume). dissimilarity_from_weights() below always consumes
src/weights.py's dyadic_weights()/tii_weights_from_*() output, so this
class of drift can't reappear.

Bug M2 fix
----------
MDSDyadicResults.py's Procrustes step had no weight matrix at all (A =
(Y_known-c_y).T @ (X_known-c_x)), while every LOO/training script -- and
the paper's own Sec. 2.1.2 ("the same weighting and alignment procedure is
applied" for the final prediction) -- uses a target-specific weighted
Procrustes. weighted_procrustes() below is the single implementation used
everywhere; there is no unweighted code path in any experiment script, so
this bug can't quietly reappear in one script while fixed in another.
"""

import numpy as np
from sklearn.manifold import MDS


def resample_poisson_symmetric(base_matrix, seed):
    """Poisson-resample an already-symmetric base matrix (e.g.
    data.load_modern_symmetric_matrix()), drawing exactly ONE independent
    sample per undirected pair (upper triangle) and mirroring it, so the
    result is symmetric by construction.

    Bug M6 fix (found by actually running mds_ancient_loo_dyadic.py, not by
    reading): src/train.py:resample_poisson() resamples every cell
    independently. Given an already-symmetric base matrix, it has no way to
    know that cell (i,j) and cell (j,i) are "the same" trade edge and must
    move together -- it draws two independent Poisson samples from the same
    lambda, which are generally NOT equal to each other. sklearn's
    MDS(dissimilarity='precomputed') raises ValueError on any asymmetry (even
    a single cell).

    Still needed for weights.symmetric_tii_from_symmetric() (modern TII),
    which requires an already-presymmetrized input and does not
    self-symmetrize. NO LONGER needed for weights.dyadic_weights() or
    weights.symmetric_tii_from_directed() (ancient TII) -- both were fixed/
    always were self-symmetrizing from a RAW (directed) input via their own
    Eq. 3 / Eq. 5-6 cross-term formulas, so plain train.resample_poisson()
    on the raw matrix is safe for both (see weights.py's dyadic-formula
    correction note; the previous version of this docstring described
    dyadic_weights() as needing a presymmetrized input -- that was only true
    of the old, incorrect implementation).

    This also matches the legacy MDS scripts' own resampling more precisely
    than train.resample_poisson() would: they drew one Poisson sample per
    row of a one-row-per-undirected-edge trade table (e.g.
    MDSDyadicBootstrap.py's `b_traffic['Traffic'] = np.random.poisson(...)`),
    never two independent samples for the same edge.

    NOTE: GCN's own modern-TII pipeline (src/evaluate.py:run_modern_loo
    ('tii')) has the same precondition violation (feeds train.resample_
    poisson() a pre-symmetrized base matrix into tii_weights_from_
    symmetric(), which assumes but doesn't enforce a symmetric input).
    GCN's directed-graph architecture doesn't crash on the resulting
    few-cell asymmetry, so it was never surfaced there. Deliberately NOT
    fixed here -- that's GCN's already-validated, already-reported
    pipeline; changing it without the user's explicit go-ahead would
    violate "ask before changing anything
    that alters a reported number." Flagged in CHANGELOG_AND_HANDOFF.md and
    to the user directly instead."""
    rng = np.random.default_rng(int(seed))
    base_matrix = np.asarray(base_matrix)
    n = base_matrix.shape[0]
    resampled = np.zeros_like(base_matrix, dtype=float)
    iu = np.triu_indices(n, k=1)
    draws = rng.poisson(base_matrix[iu]).astype(float)
    resampled[iu] = draws
    resampled[iu[1], iu[0]] = draws
    return resampled


def dissimilarity_from_weights(weight_matrix, eps=1e-9):
    """Paper Eq. 4/7: delta_ij = sqrt(1 / I_ij), with a pseudo-count eps
    added to the denominator (paper's own 1e-9) to avoid division by zero
    on non-edges, diagonal forced to 0."""
    dissim = np.sqrt(1.0 / (np.asarray(weight_matrix, dtype=float) + eps))
    np.fill_diagonal(dissim, 0.0)
    return dissim


def smacof_embed(dissimilarity_matrix, random_state):
    """Metric MDS via SMACOF (paper Eq. 8), matching every legacy script's
    call exactly: 2 components, a precomputed dissimilarity matrix, and a
    single random initialization (n_init=1, i.e. one SMACOF run per
    replicate -- the Poisson-perturbation bootstrap itself is what supplies
    randomness across replicates, not multiple restarts within one)."""
    embedding = MDS(n_components=2, dissimilarity='precomputed',
                     random_state=int(random_state), n_init=1)
    return embedding.fit_transform(dissimilarity_matrix)


def weighted_procrustes(known_real, known_mds, target_mds_point, w_vec=None):
    """Paper Sec. 2.1.2's target-specific weighted Procrustes alignment.

    known_real: (n_anchor, 2) array of [long, lat] real-world anchor coords.
    known_mds:  (n_anchor, 2) array of the same anchors' MDS coordinates,
                row-aligned with known_real.
    target_mds_point: (2,) MDS coordinate of the one city being aligned.
    w_vec: (n_anchor,) raw (not yet normalized) anchor weights -- the
        target-to-anchor dyadic share or TII value (paper's w~_j = I_tj),
        row-aligned with known_real/known_mds. Min-max normalized here
        exactly as the paper specifies: w_j = (w~_j - min)/(max - min +
        1e-9) + 0.1. If None, every anchor gets equal weight (plain,
        unweighted Procrustes) -- kept only so this function can serve as
        the ground truth in a synthetic-equivalence test; no experiment
        script in this package calls it without w_vec, since doing so is
        exactly Bug M2.

    Returns the (2,) predicted [long, lat] for the target city.
    """
    known_real = np.asarray(known_real, dtype=float)
    known_mds = np.asarray(known_mds, dtype=float)
    target_mds_point = np.asarray(target_mds_point, dtype=float)

    if w_vec is None:
        w = np.ones(known_real.shape[0])
    else:
        w = np.asarray(w_vec, dtype=float)
        w = (w - w.min()) / (w.max() - w.min() + 1e-9) + 0.1

    c1 = np.average(known_real, axis=0, weights=w)
    c2 = np.average(known_mds, axis=0, weights=w)
    A = (known_real - c1).T @ np.diag(w) @ (known_mds - c2)
    U, _, Vt = np.linalg.svd(A)
    R = Vt.T @ U.T
    scale = np.linalg.norm(known_real - c1) / np.linalg.norm(known_mds - c2)

    return (target_mds_point - c2) @ R.T * scale + c1
