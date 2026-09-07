# -*- coding: utf-8 -*-
"""
Hyperparameter search for the modern-cities Dyadic model.
Corresponds to legacy/GCNDyadicRealHyper.py. Same search grid.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, grid_search  # noqa: E402

LR_CHOICES = [0.00951, 0.0095, 0.00949]
EMBEDDING_CHOICES = [8]
MAX_EPOCHS = 2100


def main():
    val_hist, train_hist, best_config, best_epoch, min_error = grid_search.run_modern_dyadic_search(
        LR_CHOICES, EMBEDDING_CHOICES, MAX_EPOCHS,
    )
    best_lr, best_emb = best_config
    print(f"\nWINNER (Modern Dyadic): LR={best_lr}, Emb={best_emb}, Epoch={best_epoch}, Error={min_error:.2f} km")

    fig_dir = os.path.join(config.FIGURES_DIR, "grid_search_modern_dyadic")
    grid_search.plot_heatmap(val_hist, LR_CHOICES, EMBEDDING_CHOICES, "Modern Dyadic Grid Search",
                              save_path=os.path.join(fig_dir, "heatmap.png"))
    grid_search.plot_learning_curve(val_hist, train_hist, best_config, best_epoch,
                                     f"Modern Dyadic Learning Curve\n(LR={best_lr}, Emb={best_emb})",
                                     save_path=os.path.join(fig_dir, "learning_curve.png"))
    print(f"Figures saved under: {fig_dir}")
    return best_config, best_epoch, min_error


if __name__ == "__main__":
    main()
