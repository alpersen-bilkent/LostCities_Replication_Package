# -*- coding: utf-8 -*-
"""
Single training routine used by every experiment (LOO fold or final
prediction). Two bugs live here:

Bug 4 fix
---------
sklearn's MinMaxScaler inverse transform is (X - min_) / scale_. The
original scripts all wrote X / scale_ + min_, which introduces a
constant offset (~37 degrees in longitude). It happened to cancel out
of the reported LOO error because the identical wrong map was applied
to both the prediction and the target inside the loss -- but it would
corrupt anything that needs absolute coordinates, which the §7.1
directional constraints will.

Bug 3 fix
---------
The original scripts called np.random.seed(seed) *inside* the LOO
loop while the Poisson resample happened in the *outer* loop, so
iteration k's data perturbation was actually determined by iteration
k-1's last inner seed rather than by its own seed. resample_poisson()
below uses an explicit, independently-seeded generator instead of
mutating global numpy random state, and is kept separate from
torch.manual_seed (model initialization).
"""

import numpy as np
import torch

from .model import GCN
from . import config


def resample_poisson(base_matrix, seed):
    rng = np.random.default_rng(int(seed))
    return rng.poisson(base_matrix).astype(float)


def calculate_distance_km(p1_long, p1_lat, p2_long, p2_lat,
                           latitude_param=config.LATITUDE_PARAM):
    cos_factor = np.cos(np.radians(latitude_param))
    y_diff = p1_lat - p2_lat
    x_diff = p1_long - p2_long
    return config.KM_PER_DEGREE * np.sqrt(y_diff ** 2 + (cos_factor * x_diff) ** 2)


def _km_loss(pred_scaled, target_scaled_subset, min_v, scale_v, cos_l):
    """The one place the (bug-4-fixed) distance loss is computed, so
    fit_predict() and fit_with_curve() below can never drift apart the
    way the original scripts' copy-pasted loss formulas did."""
    p_un = (pred_scaled - min_v) / scale_v
    t_un = (target_scaled_subset - min_v) / scale_v
    d_sq = (t_un[:, 1] - p_un[:, 1]) ** 2 + (cos_l * (t_un[:, 0] - p_un[:, 0])) ** 2
    return torch.mean(config.KM_PER_DEGREE * torch.sqrt(d_sq + 1e-6))


def fit_predict(graph_data, num_nodes, embedding_dim, epochs, lr, seed,
                 target_scaled, scaler, supervised_node_idx, train_mask_local,
                 predict_node_idx, latitude_param=config.LATITUDE_PARAM,
                 constraint_fn=None, gamma=0.0, weight_decay=5e-4):
    """Trains one GCN and returns unscaled [long, lat] predictions at
    predict_node_idx.

    graph_data:          full-graph Data object (all nodes, incl. any
                          lost/held-out cities as message-passing nodes).
    supervised_node_idx: node indices (into graph_data) that have ground
                          truth, row-aligned with target_scaled.
    train_mask_local:    positions into supervised_node_idx / target_scaled
                          actually used for the loss this run (a LOO fold
                          excludes the held-out city here, not from the
                          graph).
    predict_node_idx:    node indices (into graph_data) to return
                          predictions for once training is done.
    constraint_fn:       optional callable(all_nodes_unscaled) -> scalar km
                          penalty (see src/constraints.py). Adds
                          gamma * constraint_fn(...) to the loss every
                          epoch, on ALL node predictions (not just
                          train_nodes) -- §7.1's "one model, one loss,
                          always". None/gamma=0 (the default) reproduces
                          the unconstrained loss exactly, so every
                          existing experiment is unaffected.
    weight_decay:        Adam's L2 penalty. Default 5e-4 matches every
                          existing script's behavior exactly (this used
                          to be a hardcoded literal here, inherited
                          unchanged from the legacy scripts and never
                          actually tuned) -- only diagnostic scripts that
                          explicitly pass a different value are affected.
    """
    torch.manual_seed(int(seed))

    model = GCN(num_nodes=num_nodes, embedding_dim=embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    min_v = torch.tensor(scaler.min_, dtype=torch.float)
    scale_v = torch.tensor(scaler.scale_, dtype=torch.float)
    cos_l = np.cos(np.radians(latitude_param))

    train_nodes = supervised_node_idx[train_mask_local]
    train_target = target_scaled[train_mask_local]

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(graph_data)
        loss = _km_loss(out[train_nodes], train_target, min_v, scale_v, cos_l)

        if constraint_fn is not None and gamma:
            all_unscaled = (out - min_v) / scale_v
            loss = loss + gamma * constraint_fn(all_unscaled)

        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        final_out = model(graph_data)
        preds = scaler.inverse_transform(final_out[predict_node_idx].numpy())
    return preds


def fit_with_curve(graph_data, num_nodes, embedding_dim, epochs, lr, seed,
                    target_scaled, scaler, supervised_node_idx, train_mask_local,
                    val_mask_local, latitude_param=config.LATITUDE_PARAM):
    """Like fit_predict(), but returns per-epoch (train_curve, val_curve)
    KM-error arrays instead of a final prediction. Used only by the
    hyperparameter grid search (src/grid_search.py) to choose epoch count,
    learning rate and embedding dimension -- never to produce a reported
    LOO or prediction number itself. No Poisson resampling here (the
    search is deterministic, single-run-per-fold), so Bug 3 does not
    apply to this function."""
    torch.manual_seed(int(seed))

    model = GCN(num_nodes=num_nodes, embedding_dim=embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    min_v = torch.tensor(scaler.min_, dtype=torch.float)
    scale_v = torch.tensor(scaler.scale_, dtype=torch.float)
    cos_l = np.cos(np.radians(latitude_param))

    train_nodes = supervised_node_idx[train_mask_local]
    train_target = target_scaled[train_mask_local]
    val_nodes = supervised_node_idx[val_mask_local]
    val_target = target_scaled[val_mask_local]

    train_curve = np.zeros(epochs)
    val_curve = np.zeros(epochs)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(graph_data)
        loss = _km_loss(out[train_nodes], train_target, min_v, scale_v, cos_l)
        loss.backward()
        optimizer.step()
        train_curve[epoch] = loss.item()

        model.eval()
        with torch.no_grad():
            out_eval = model(graph_data)
            val_err = _km_loss(out_eval[val_nodes], val_target, min_v, scale_v, cos_l)
            val_curve[epoch] = val_err.item()

    return train_curve, val_curve
