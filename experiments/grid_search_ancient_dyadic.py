# -*- coding: utf-8 -*-
"""
Hyperparameter search for the ancient-network Dyadic model.
Corresponds to legacy/GNNHyperDyadic.py, with Bug 2 fixed (searches the
full 25-node graph instead of the 15-known-city-only graph the original
search used). Same search grid as the original -- narrow, centered on
the values already in use, because it was a fine-tuning pass.

Because the graph changed, this search's winner is not guaranteed to
match the EPOCHS=1473 / LR=0.0053 / EMBEDDING_DIM=16 currently used in
experiments/ancient_loo_dyadic.py and experiments/ancient_lost_dyadic.py.
If the winner found here differs, do NOT copy it into those experiment
configs without confirming first -- that would change reported numbers
beyond the direct bug fixes. If the new winner sits at the edge of this
narrow grid, widen LR_CHOICES and rerun before trusting it.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, grid_search  # noqa: E402

LR_CHOICES = [0.0052, 0.00525, 0.0053, 0.00535]
EMBEDDING_CHOICES = [16]
MAX_EPOCHS = 2100


def main():
    val_hist, train_hist, best_config, best_epoch, min_error = grid_search.run_ancient_dyadic_search(
        LR_CHOICES, EMBEDDING_CHOICES, MAX_EPOCHS,
    )
    best_lr, best_emb = best_config
    print(f"\nWINNER (Ancient Dyadic, 25-node graph): LR={best_lr}, Emb={best_emb}, "
          f"Epoch={best_epoch}, Error={min_error:.2f} km")
    print("Compare against the currently hardcoded EPOCHS=1473, LR=0.0053, EMBEDDING_DIM=16 "
          "before changing anything -- see module docstring.")

    fig_dir = os.path.join(config.FIGURES_DIR, "grid_search_ancient_dyadic")
    grid_search.plot_heatmap(val_hist, LR_CHOICES, EMBEDDING_CHOICES, "Ancient Dyadic Grid Search (25-node graph)",
                              save_path=os.path.join(fig_dir, "heatmap.png"))
    grid_search.plot_learning_curve(val_hist, train_hist, best_config, best_epoch,
                                     f"Ancient Dyadic Learning Curve\n(LR={best_lr}, Emb={best_emb})",
                                     save_path=os.path.join(fig_dir, "learning_curve.png"))
    print(f"Figures saved under: {fig_dir}")
    return best_config, best_epoch, min_error


if __name__ == "__main__":
    main()
