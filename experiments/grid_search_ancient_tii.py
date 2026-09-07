# -*- coding: utf-8 -*-
"""
Hyperparameter search for the ancient-network TII model.

No legacy hyperparameter-search script exists for this one (only
GCNDyadicRealHyper.py, GCNTIIRealHyper.py and GNNHyperDyadic.py were
provided) -- legacy/GCNTIIBootstrap.py's EPOCHS=244/LR=0.011/
EMBEDDING_DIM=8 have no known search behind them, and are additionally
subject to Bug 2 (trained on the old 15-node-only graph). So this grid
is intentionally wider than grid_search_ancient_dyadic.py's (which had
a real prior search to fine-tune around) -- treat this as a first pass;
if the winner sits at an edge, widen and rerun.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, grid_search  # noqa: E402

LR_CHOICES = [0.005, 0.008, 0.011, 0.014, 0.017]
EMBEDDING_CHOICES = [8, 16]
MAX_EPOCHS = 800


def main():
    val_hist, train_hist, best_config, best_epoch, min_error = grid_search.run_ancient_tii_search(
        LR_CHOICES, EMBEDDING_CHOICES, MAX_EPOCHS,
    )
    best_lr, best_emb = best_config
    print(f"\nWINNER (Ancient TII, 25-node graph): LR={best_lr}, Emb={best_emb}, "
          f"Epoch={best_epoch}, Error={min_error:.2f} km")
    print("Compare against the currently hardcoded EPOCHS=244, LR=0.011, EMBEDDING_DIM=8 "
          "before changing anything -- see module docstring.")

    fig_dir = os.path.join(config.FIGURES_DIR, "grid_search_ancient_tii")
    grid_search.plot_heatmap(val_hist, LR_CHOICES, EMBEDDING_CHOICES, "Ancient TII Grid Search (25-node graph)",
                              save_path=os.path.join(fig_dir, "heatmap.png"))
    grid_search.plot_learning_curve(val_hist, train_hist, best_config, best_epoch,
                                     f"Ancient TII Learning Curve\n(LR={best_lr}, Emb={best_emb})",
                                     save_path=os.path.join(fig_dir, "learning_curve.png"))
    print(f"Figures saved under: {fig_dir}")
    return best_config, best_epoch, min_error


if __name__ == "__main__":
    main()
