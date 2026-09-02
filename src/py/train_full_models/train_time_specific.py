# train_time_specific.py

"""
Trains one binary classifier for a single (model, tissue, window) - the
time-specific training mode.

Inputs:
  - data/training/hrs<window>/data_diff_hrs<window>.csv (motif enrichment features)
  - data/training/hrs<window>/y_<tissue>.csv (binary loop-presence labels)

Outputs:
  - results/train_full_models/train_time_specific/models/cv/<model>/hrs<window>/<tissue>.pkl
  - results/train_full_models/train_time_specific/models/all_data/<model>/hrs<window>/<tissue>.pkl
  - results/train_full_models/train_time_specific/figures/<model>/hrs<window>/<tissue>.{png,pdf}
  - results/train_full_models/train_time_specific/roc_results/<model>/hrs<window>/<tissue>.pkl
  - results/train_full_models/train_time_specific/cv_aucroc_summary/<model>/hrs<window>/<tissue>.csv
"""

import argparse
import logging
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

from core.constants import MODELS, TIME_SPECIFIC_MODEL_PARAMS, WINDOWS
from core.log import configure_logging
from core.paths import (
    train_time_specific_figure_dir,
    train_time_specific_model_path,
    train_time_specific_roc_result_path,
    train_time_specific_summary_path,
)
from train_full_models.cv import cross_validate, fit_full_data_model
from train_full_models.io import load_time_specific_features, save_model, save_cv_result, save_summary_row
from train_full_models.plotting import plot_roc

logger = logging.getLogger(__name__)


def run_time_specific(model: str, tissue: str, window: str, n_splits: int) -> dict:
    """
    Cross-validate and fit a single classifier for one (model, tissue, window).

    Returns:
        Summary dict with tissue, model, window, and CV metrics.
    """
    classifier = MODELS[model]["class"](**TIME_SPECIFIC_MODEL_PARAMS[model])
    full = MODELS[model]["full"]

    logger.info(f"Time-specific training: {full} | {tissue} | hrs{window}")

    X, y = load_time_specific_features(tissue, window)

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    result = cross_validate(classifier, X, y, splitter, stratify_target=None)

    fig_dir = train_time_specific_figure_dir(model, window)
    plot_roc(
        result,
        title=(
            f"{full} ROC - {tissue}, hrs{window}\n"
            f"AUC = {result.mean_auc:.3f} ± {result.std_auc:.3f}\n"
            f"Acc = {result.mean_acc:.3f} ± {result.std_acc:.3f}"
        ),
        out_paths=[
            Path(fig_dir) / f"{tissue}.{fmt}"
            for fmt in ("png", "pdf")
        ],
    )

    # save the model trained on the single best CV fold (highest AUC),
    best_fold = int(np.argmax(result.roc_aucs))
    train_idx, _ = result.fold_indices[best_fold]
    best_clf = fit_full_data_model(classifier, X.iloc[train_idx], y.iloc[train_idx])
    save_model(best_clf, train_time_specific_model_path("cv", model, window, tissue))

    all_clf = fit_full_data_model(classifier, X, y)
    save_model(all_clf, train_time_specific_model_path("all_data", model, window, tissue))

    # so combined_roc_plot.py can overlay this tissue's ROC curve with the other two, without re-running CV
    save_cv_result(result, train_time_specific_roc_result_path(model, window, tissue))

    summary_row = {
        "tissue": tissue,
        "model": model,
        "window": window,
        "mean_auc": round(result.mean_auc, 6),
        "std_auc": round(result.std_auc, 6),
        "mean_acc": round(result.mean_acc, 6),
        "std_acc": round(result.std_acc, 6),
    }

    summary_path = train_time_specific_summary_path(model, window, tissue)
    save_summary_row(summary_row, summary_path)
    logger.info(f"Saved summary: {summary_path}")

    logger.info(f"{tissue} hrs{window} done")

    return summary_row


def main() -> None:
    parser = argparse.ArgumentParser(description="Time-specific classifier training")
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--tissue", required=True, help="Tissue name, e.g. Glia")
    parser.add_argument("--window", required=True, choices=WINDOWS)
    parser.add_argument("--n_splits", type=int, required=True)
    parser.add_argument("--log_path", type=Path, required=True)
    args = parser.parse_args()

    configure_logging(args.log_path)
    run_time_specific(args.model, args.tissue, args.window, args.n_splits)
    logger.info("Done.")


if __name__ == "__main__":
    main()
