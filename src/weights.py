# -*- coding: utf-8 -*-
"""
Edge-weight schemes. This is the ONLY place either weighting scheme is
defined, so "did training and prediction use the same edge weights?"
is answerable by reading this one file.

Bug 1 fix
---------
The original GCNDyadicBootstrap.py (training/LOO, paper Table 9) and
GCNDyadicBootstrapResults.py (final lost-city prediction, paper Table 11)
computed two different formulas under the same "dyadic share" name:
training divided by the symmetrized row total V_i (paper Eq. 2-3);
prediction divided by the raw column total (imports only) of the
destination city. dyadic_weights() below is the single definition now
used by both.

Dyadic-formula correction, reverted (checked against the primary source)
--------------------------------------------------------------------------
A later change (same day, MDS session) switched this function from a
presymmetrized-numerator formula to a directional one
(w_ij = 0.5*(X_ij/V_i + X_ji/V_j), each direction's own raw count divided
by its own source city's total), reasoning that this matches the paper's
Eq. 3 as literally typeset and that Barbieri/Gartzke & Li's cited formula
is "directional by construction." Checked directly against the primary
source (Gartzke & Li 2003, Table I) and reverted: Barbieri's own
trade_share_i is (imports_ij + exports_ij)/(imports_i + exports_i) -- the
numerator is imports PLUS exports, i.e. the TOTAL bilateral volume between
i and j (both directions summed), used identically in both i's and j's
share calculation; only the denominator (each state's own total trade)
differs between the two. "Directional" in that literature means
trade_share_i != trade_share_j because of the denominator, not because the
numerator is a one-way flow -- her own trade_symmetry measure exists to
compare exactly the two full-bilateral-volume shares against each other.
The paper's own prose immediately above Eq. 3 says the same thing
Barbieri's formula does ("the share of total trade volume BETWEEN THEM"),
which does not match Eq. 3's literal one-directional notation either --
Eq. 3 appears to have a typo in the manuscript, not a deliberate
adaptation; worth a small correction there (X_ij+X_ji in place of the bare
X_ij/X_ji terms) so the equation matches both its own citation and its own
surrounding text. The ~123 vs ~126 km empirical difference cited as
supporting evidence for the directional version is not relevant either
way -- a formula scoring marginally better on LOO error doesn't establish
which one correctly implements a cited definition; those are separate
questions. dyadic_weights() below keeps the interface improvement (takes a
RAW, possibly directional matrix and self-symmetrizes internally, so
callers no longer need to presymmetrize first) but restores the
total-bilateral-volume formula in the computation itself. This is shared
code -- the fix applies to GCN and MDS identically.

Bug 5 fix
---------
The original TII scripts min-max normalized log(1+TII) over the WHOLE
matrix including zero (non-edge) cells, which maps the minimum
existing edge to exactly 0.0 and then create_graph_data's `> 0` filter
silently drops it. _lognorm_edges() below determines which cells are
edges BEFORE normalizing, and only rescales those cells, so the
weakest real edge is never zeroed out.
"""

import numpy as np


def dyadic_weights(raw_matrix):
    """Dyadic trade share, paper Eq. 2-3 (corrected against Barbieri's own
    published formula -- see this module's docstring): V_i = sum_k X_ik +
    sum_k X_ki (Eq. 2, total trade volume for city i, both directions);
    w_ij = 0.5*(vol_ij/V_i + vol_ij/V_j) (Eq. 3), where vol_ij = X_ij+X_ji
    is the TOTAL bilateral volume between i and j (both directions summed,
    matching Barbieri's imports_ij+exports_ij), used identically in both
    terms -- only the denominator differs. `raw_matrix` is the DIRECTED
    matrix (X_ij need not equal X_ji) -- do not presymmetrize before
    calling this; the function symmetrizes internally, so the output is
    exactly symmetric regardless of the input's asymmetry."""
    raw_matrix = np.asarray(raw_matrix, dtype=float)
    sym_matrix = raw_matrix + raw_matrix.T
    x_out = raw_matrix.sum(axis=1)
    x_in = raw_matrix.sum(axis=0)
    total_volumes = x_out + x_in
    total_volumes_safe = np.where(total_volumes == 0, 1.0, total_volumes)

    weights = np.zeros_like(raw_matrix, dtype=float)
    rows, cols = np.where(sym_matrix > 0)
    vol = sym_matrix[rows, cols]
    weights[rows, cols] = 0.5 * (
        vol / total_volumes_safe[rows] + vol / total_volumes_safe[cols]
    )
    return weights


def _lognorm_edges(sym_matrix):
    """log1p then min-max normalize, computed only over existing edges."""
    edge_mask = sym_matrix > 0
    normalized = np.zeros_like(sym_matrix, dtype=float)
    if not edge_mask.any():
        return normalized

    log_vals = np.log1p(sym_matrix)
    edge_vals = log_vals[edge_mask]
    v_min, v_max = edge_vals.min(), edge_vals.max()
    if v_max > v_min:
        normalized[edge_mask] = (log_vals[edge_mask] - v_min) / (v_max - v_min)
    else:
        normalized[edge_mask] = log_vals[edge_mask]
    np.fill_diagonal(normalized, 0)
    return normalized


def symmetric_tii_from_symmetric(symmetric_matrix):
    """The raw (pre-log-normalization) symmetrized TII score, paper Eq. 5-6,
    on an already-undirected matrix (used for the modern network, where the
    source spreadsheet is symmetrized on load). Exposed separately from
    tii_weights_from_symmetric() below because GCN's Eq. 9 log+min-max
    transform is GCN-specific (Sec. 2.2.1: it exists to stop the neural
    net's edge weights from being squashed by TII's heavy-tailed range) --
    MDS's own TII dissimilarity (Eq. 7, delta_ij = sqrt(1/T'_ij)) uses this
    raw score directly, with no log transform. src/mds.py consumes this
    function; do not point MDS at tii_weights_from_symmetric()'s log-
    normalized output, or its dissimilarity input silently stops matching
    the paper's own MDS methodology."""
    city_totals = symmetric_matrix.sum(axis=1)
    world_total = symmetric_matrix.sum()
    expected = np.outer(city_totals, city_totals)
    expected[expected == 0] = 1e-9
    return (symmetric_matrix * world_total) / expected


def symmetric_tii_from_directed(raw_matrix):
    """The raw (pre-log-normalization) symmetrized TII score, paper Eq. 5-6,
    on directed itinerary counts (used for the ancient network), normalizing
    exports and imports separately before symmetrizing. See
    symmetric_tii_from_symmetric()'s docstring above for why this is exposed
    separately from tii_weights_from_directed()'s GCN-specific log-normalized
    output -- MDS's src/mds.py consumes this one directly."""
    x_out = raw_matrix.sum(axis=1)
    x_in = raw_matrix.sum(axis=0)
    world_total = raw_matrix.sum()

    with np.errstate(divide='ignore', invalid='ignore'):
        out_in = np.outer(x_out, x_in)
        in_out = np.outer(x_in, x_out)
        T_ij = np.where(out_in > 0, (raw_matrix * world_total) / out_in, 0.0)
        T_ji = np.where(in_out > 0, (raw_matrix.T * world_total) / in_out, 0.0)

    return 0.5 * (T_ij + T_ji)


def tii_weights_from_symmetric(symmetric_matrix):
    """GCN edge weights (paper Eq. 9): TII on an already-undirected matrix,
    log+min-max normalized so the neural net doesn't ignore small-but-real
    links next to a heavy-tailed TII distribution."""
    return _lognorm_edges(symmetric_tii_from_symmetric(symmetric_matrix))


def tii_weights_from_directed(raw_matrix):
    """GCN edge weights (paper Eq. 9): TII on directed itinerary counts,
    log+min-max normalized (see tii_weights_from_symmetric()'s docstring)."""
    return _lognorm_edges(symmetric_tii_from_directed(raw_matrix))
