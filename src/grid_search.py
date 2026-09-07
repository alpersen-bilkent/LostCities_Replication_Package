# -*- coding: utf-8 -*-
"""
"Peak-potential" hyperparameter grid search: for each (learning rate,
embedding dimension) combination, trains one deterministic LOO model
per known city (no Poisson resampling -- this is a diagnostic search,
not a reported result) and records the average validation-error curve
across folds, epoch by epoch. The winning config is whichever curve
ever reaches the lowest average validation error; the winning epoch is
where that curve is lowest.

This collapses legacy/GCNDyadicRealHyper.py, legacy/GCNTIIRealHyper.py
and legacy/GNNHyperDyadic.py into one driver. Bug 4 (wrong MinMaxScaler
inverse) is fixed by going through src/train.py's shared loss function.
Bug 5 (TII dropping its weakest edge) is fixed by going through
src/weights.py. legacy/GNNHyperDyadic.py additionally had Bug 2 (its
ancient network graph was restricted to the 15 known cities, silently
dropping the 10 lost cities as graph nodes) -- run_ancient_dyadic_search()
below uses the full 25-node graph instead, exactly like
src/evaluate.py:run_ancient_loo().

IMPORTANT: because the ancient search here runs on a materially
different graph than the one legacy/GNNHyperDyadic.py searched, its
winning (lr, embedding_dim, epoch) is not guaranteed to match the
EPOCHS=1473 / LR=0.0053 / EMBEDDING_DIM=16 currently hardcoded into
experiments/ancient_loo_dyadic.py and experiments/ancient_lost_dyadic.py.
Re-running this search and adopting a new winner would change numbers
downstream of it -- do not update those experiment configs from a new
search result without confirming first.
"""

import os

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

from . import config, data, train, weights

try:
    import matplotlib
    if os.environ.get("GCN_HEADLESS"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_PLT = True
except ImportError:
    _HAS_PLT = False


def run_loo_grid_search(num_nodes, base_matrix, weight_fn, known_node_idx, known_coords,
                         lr_choices, embedding_choices, max_epochs,
                         base_seed=config.MASTER_SEED, desc="Grid search"):
    """Returns (val_history, train_history, best_config, best_epoch, min_error).
    Both histories map (lr, embedding_dim) -> average error curve (length
    max_epochs, in km), across the LOO folds. Captured in the same pass so
    the winning config never needs a separate "rerun to get the training
    curve" step, unlike the original scripts."""
    weight_matrix = weight_fn(base_matrix)
    graph_data = data.create_graph_data(weight_matrix, num_nodes)

    scaler = MinMaxScaler().fit(known_coords)
    target_scaled = torch.tensor(scaler.transform(known_coords), dtype=torch.float)
    known_node_idx = np.asarray(known_node_idx)
    num_known = len(known_node_idx)

    val_history, train_history = {}, {}
    for embedding_dim in embedding_choices:
        for lr in lr_choices:
            print(f"{desc}: embedding_dim={embedding_dim}, lr={lr}")
            agg_val_curve = np.zeros(max_epochs)
            agg_train_curve = np.zeros(max_epochs)

            for i in range(num_known):
                train_mask_local = np.array([j for j in range(num_known) if j != i])
                val_mask_local = np.array([i])
                train_curve, val_curve = train.fit_with_curve(
                    graph_data, num_nodes, embedding_dim, max_epochs, lr, base_seed,
                    target_scaled, scaler, known_node_idx, train_mask_local, val_mask_local,
                )
                agg_val_curve += val_curve
                agg_train_curve += train_curve

            agg_val_curve /= num_known
            agg_train_curve /= num_known
            val_history[(lr, embedding_dim)] = agg_val_curve
            train_history[(lr, embedding_dim)] = agg_train_curve

    best_config, best_epoch, min_error = pick_winner(val_history)
    return val_history, train_history, best_config, best_epoch, min_error


def pick_winner(val_history):
    best_config, min_error = None, float('inf')
    for config_key, curve in val_history.items():
        peak = curve.min()
        if peak < min_error:
            min_error = peak
            best_config = config_key
    best_epoch = int(np.argmin(val_history[best_config]))
    return best_config, best_epoch, min_error


def run_modern_dyadic_search(lr_choices, embedding_choices, max_epochs,
                              base_seed=config.MASTER_SEED):
    """Corresponds to legacy/GCNDyadicRealHyper.py. No Bug 2 (no lost
    cities in the modern network)."""
    valid_cities, known_coords = data.modern_cities_and_coords()
    num_nodes = len(valid_cities)
    # dyadic_weights() self-symmetrizes from a RAW (directed) matrix (see
    # weights.py) -- use the raw loader, not load_modern_symmetric_matrix().
    base_matrix = data.load_modern_matrix(valid_cities=valid_cities)
    known_idx = np.arange(num_nodes)

    return run_loo_grid_search(
        num_nodes, base_matrix, weights.dyadic_weights, known_idx, known_coords,
        lr_choices, embedding_choices, max_epochs, base_seed, desc="Modern Dyadic grid search",
    )


def run_modern_tii_search(lr_choices, embedding_choices, max_epochs,
                           base_seed=config.MASTER_SEED):
    """Corresponds to legacy/GCNTIIRealHyper.py. No Bug 2. Bug 5 fixed via
    weights.tii_weights_from_symmetric()."""
    valid_cities, known_coords = data.modern_cities_and_coords()
    num_nodes = len(valid_cities)
    base_matrix = data.load_modern_symmetric_matrix(valid_cities=valid_cities)
    known_idx = np.arange(num_nodes)

    return run_loo_grid_search(
        num_nodes, base_matrix, weights.tii_weights_from_symmetric, known_idx, known_coords,
        lr_choices, embedding_choices, max_epochs, base_seed, desc="Modern TII grid search",
    )


def run_ancient_dyadic_search(lr_choices, embedding_choices, max_epochs,
                               base_seed=config.MASTER_SEED):
    """Corresponds to legacy/GNNHyperDyadic.py, with Bug 2 fixed: searches
    on the full 25-node ancient graph (lost cities present as
    message-passing nodes) instead of the original's 15-known-city-only
    graph. See module docstring -- this can change the winning config
    relative to the original search."""
    known_idx, _, _, _ = data.ancient_known_lost_split()
    known_coords, _ = data.ancient_known_coords()
    num_nodes = len(data.ANCIENT_CITY_NAMES)
    # dyadic_weights() self-symmetrizes from a RAW (directed) matrix (see
    # weights.py) -- use ANCIENT_MATRIX_DATA directly, not the presymmetrized
    # ancient_symmetric_matrix().
    base_matrix = data.ANCIENT_MATRIX_DATA

    return run_loo_grid_search(
        num_nodes, base_matrix, weights.dyadic_weights, known_idx, known_coords,
        lr_choices, embedding_choices, max_epochs, base_seed, desc="Ancient Dyadic grid search",
    )


def run_ancient_tii_search(lr_choices, embedding_choices, max_epochs,
                            base_seed=config.MASTER_SEED):
    """No legacy hyperparameter-search script existed for ancient TII (only
    GCNDyadicRealHyper.py, GCNTIIRealHyper.py and GNNHyperDyadic.py were
    provided) -- legacy/GCNTIIBootstrap.py's EPOCHS=244/LR=0.011/
    EMBEDDING_DIM=8 have no known search behind them at all, on top of
    also being subject to Bug 2 (same 15-node-only graph problem as the
    dyadic search). Searches the full 25-node ancient graph, matching
    run_ancient_dyadic_search()."""
    known_idx, _, _, _ = data.ancient_known_lost_split()
    known_coords, _ = data.ancient_known_coords()
    num_nodes = len(data.ANCIENT_CITY_NAMES)
    base_matrix = data.ANCIENT_MATRIX_DATA

    return run_loo_grid_search(
        num_nodes, base_matrix, weights.tii_weights_from_directed, known_idx, known_coords,
        lr_choices, embedding_choices, max_epochs, base_seed, desc="Ancient TII grid search",
    )


def plot_heatmap(grid_history, lr_choices, embedding_choices, title, save_path=None):
    if not _HAS_PLT:
        return None
    heatmap_data = np.zeros((len(embedding_choices), len(lr_choices)))
    for e_idx, emb in enumerate(embedding_choices):
        for l_idx, lr in enumerate(lr_choices):
            heatmap_data[e_idx, l_idx] = grid_history[(lr, emb)].min()

    fig = plt.figure(figsize=(8, 6))
    plt.imshow(heatmap_data, cmap='magma_r', aspect='auto')
    plt.colorbar(label='Min LOOCV Error (km)')
    plt.xticks(np.arange(len(lr_choices)), lr_choices)
    plt.yticks(np.arange(len(embedding_choices)), embedding_choices)
    plt.xlabel('Learning Rate')
    plt.ylabel('Embedding Dimension')
    plt.title(title)
    for (j, i), val in np.ndenumerate(heatmap_data):
        plt.text(i, j, f'{val:.1f}', ha='center', va='center',
                  color='white' if val > heatmap_data.mean() else 'black')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.show()
    plt.close(fig)
    return fig


def plot_learning_curve(val_history, train_history, best_config, best_epoch, title, save_path=None):
    if not _HAS_PLT:
        return None
    fig = plt.figure(figsize=(10, 6))
    plt.plot(train_history[best_config], label='Avg Training Loss (LOOCV)', color='blue', alpha=0.5)
    plt.plot(val_history[best_config], label='Avg Validation Loss (LOOCV)', color='red', linewidth=2)
    plt.axvline(best_epoch, color='green', linestyle='--', label=f'Best Epoch ({best_epoch})')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Error (km)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.show()
    plt.close(fig)
    return fig
