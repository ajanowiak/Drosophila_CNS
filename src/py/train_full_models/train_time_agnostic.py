# train_time_agnostic.py

"""
Trains one binary classifier for a single (model, tissue, feature_mode) on a
dataset stacked across all three developmental windows -- covers both the
time-agnostic and expanded time-agnostic training modes.

feature_mode selects which window's enrichment scores feed each classifier:
curr (time-agnostic), prev, or expanded (curr and prev concatenated).

Inputs:
  - results/training_data/unfiltered/hrs<window>/motif_enrichment_hrs<window>.csv
  - results/training_data/unfiltered/hrs<window>/y_<tissue>.csv

Outputs:
  - results/models/time_agnostic/<model>/<feature_mode>/<model>_<tissue>_<feature_mode>.pkl
  - results/figures/time_agnostic/<model>_<feature_mode>/roc_<model>_<feature_mode>_<tissue>.{pdf,png}
  - results/time_agnostic/<model>/<feature_mode>/cv_aucroc_summary_<model>_<feature_mode>_<tissue>.csv
"""

import argparse
import logging
from pathlib import Path

from sklearn.model_selection import StratifiedKFold

from core.config import load_config
from core.constants import MODELS, TIME_AGNOSTIC_MODEL_PARAMS, FeatureMode
from core.features import stack_windows
from core.log import configure_logging
from cv import cross_validate, fit_full_data_model
from train_full_models.io import save_model, save_summary_row
from plotting import plot_roc

logger = logging.getLogger(__name__)


def run_time_agnostic(model: str, tissue: str, feature_mode: FeatureMode, n_splits: int) -> dict:
    """
    Cross-validate and fit a single classifier for one (model, tissue, feature_mode).

    Returns:
        Summary dict with tissue, model, feature_mode, and CV metrics.
    """
    classifier = MODELS[model]["class"](**TIME_AGNOSTIC_MODEL_PARAMS[model])
    full = MODELS[model]["full"]
    mode_tag = feature_mode.value
    color = load_config()["plot_colors"]["feature_mode"][mode_tag]

    logger.info(f"Time-agnostic training: {full} | {tissue} | {mode_tag}")

    X, y, composite = stack_windows(tissue, feature_mode=feature_mode)

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    result = cross_validate(classifier, X, y, splitter, stratify_target=composite)

    fig_dir = f"results/figures/time_agnostic/{model}_{mode_tag}"
    plot_roc(
        result,
        title=(
            f"Time-agnostic {full} ROC: {mode_tag} features ({tissue})\n"
            f"AUC = {result.mean_auc:.3f} ± {result.std_auc:.3f}\n"
            f"Acc = {result.mean_acc:.3f} ± {result.std_acc:.3f}"
        ),
        out_paths=[
            Path(fig_dir) / f"roc_{model}_{mode_tag}_{tissue}.{fmt}"
            for fmt in ("pdf", "png")
        ],
        color=color,
        line_alpha=0.8,
        line_label=f"Mean ROC (AUC = {result.mean_auc:.3f} ± {result.std_auc:.3f})",
        fill_label="± 1 std. dev.",
    )
    logger.info(f"[{tissue}] ROC figure saved to {fig_dir}")

    all_clf = fit_full_data_model(classifier, X, y)
    model_dir = f"results/models/time_agnostic/{model}/{mode_tag}"
    save_model(all_clf, f"{model_dir}/{model}_{tissue}_{mode_tag}.pkl")
    logger.info(f"[{tissue}] Full-data model saved to {model_dir}")

    summary_row = {
        "tissue": tissue,
        "model": model,
        "feature_mode": mode_tag,
        "mean_auc": round(result.mean_auc, 6),
        "std_auc": round(result.std_auc, 6),
        "mean_acc": round(result.mean_acc, 6),
        "std_acc": round(result.std_acc, 6),
    }

    summary_path = f"results/time_agnostic/{model}/{mode_tag}/cv_aucroc_summary_{model}_{mode_tag}_{tissue}.csv"
    save_summary_row(summary_row, summary_path)
    logger.info(f"Saved summary: {summary_path}")

    return summary_row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Time-agnostic classifier training (curr / prev / expanded feature modes)"
    )
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--tissue", required=True, help="Tissue name, e.g. Glia")
    parser.add_argument(
        "--feature_mode", type=FeatureMode, choices=list(FeatureMode), required=True
    )
    parser.add_argument("--n_splits", type=int, required=True)
    parser.add_argument("--log_path", type=Path, required=True)
    args = parser.parse_args()

    configure_logging(args.log_path)
    run_time_agnostic(args.model, args.tissue, args.feature_mode, args.n_splits)
    logger.info("Done.")


if __name__ == "__main__":
    main()
