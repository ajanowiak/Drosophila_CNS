# human_prepare_data/run_stage1_human.py

"""
Human Stage 1 driver: one condition's loop-presence classifier.

Reuses the Drosophila CV harness (cross_validate), ROC plotting (plot_roc) and
the exact time-specific XGBoost hyperparameters. The human glue loads one
feature set's enrichment matrix (pooled motifs, or motif x cell group) and each
condition's label vector (its positives + clean negatives), applies the NaN
policy below, and reports/plots the ROC. Snakemake runs one job per condition;
summarize_stage1.py then collects the conditions of one feature set.

Inputs (results/human_prototype/<negatives>/prepare_data/<feature_set>/):
  - motif_enrichment.csv, y_<condition>.csv, n_cells.tsv
Outputs (results/human_prototype/<negatives>/train/<feature_set>/):
  - cv_<condition>.csv (one summary row), roc_results/<condition>.pkl,
    figures/roc_<condition>.{png,pdf}
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import KFold

from core.constants import MODELS, TIME_SPECIFIC_MODEL_PARAMS
from core.log import configure_logging
from human_prepare_data.constants import CONDITIONS, NAN_TOLERANT_MODELS
from train_full_models.cv import CVResult, cross_validate
from train_full_models.io import save_cv_result
from train_full_models.plotting import plot_roc

logger = logging.getLogger(__name__)


def describe_features(n_features: int, n_groups: int) -> str:
    """Feature count for plot titles, split into motifs x cell groups when grouped."""
    if n_groups == 1:
        return f"{n_features:,} motifs"
    return f"{n_features:,} features ({n_features // n_groups:,} motifs x {n_groups} cell groups)"


def condition_title(full: str, cond: str, feature_set: str, n_loops: int, features: str,
                    n_cells: int, mean_auc: float, std_auc: float) -> str:
    """Drosophila-style three-line ROC title: what/from-what, sizes, then AUC."""
    title = (
        f"{full} ROC: {cond} loops, {feature_set} features\n"
        f"{n_loops:,} loops x {features}, {n_cells:,} cells\n"
        f"AUC = {mean_auc:.3f} ± {std_auc:.3f}"
    )
    return title


def select_training_loops(enrichment: pd.DataFrame, y: pd.Series, model: str) -> tuple[pd.DataFrame, pd.Series]:
    """Feature matrix and integer labels of the loops one condition's classifier trains on.

    NaN policy: a loop with no both-open "11" cell (within a cell group, for
    grouped features) has undefined (NaN) enrichment for those features. For a
    NaN-tolerant model (XGBoost) we KEEP such loops - the classifier routes the
    missing values, and every feature set then trains on the identical loop set
    (checked in aggregate_results.py). For a model that cannot take NaN we fall
    back to dropping every row with any NaN, which for grouped features drops a
    loop as soon as one group lacks a both-open cell.
    """
    X = enrichment.loc[y.index]

    if model in NAN_TOLERANT_MODELS:
        keep = y.notna()
        n_nan = int(X.loc[keep].isna().any(axis=1).sum())
        logger.info(f"  [{y.name}] keeping {n_nan} loops with NaN enrichment "
                    f"({model} handles missing natively)")
    else:
        keep = X.notna().all(axis=1) & y.notna()

    n_drop = int((~keep).sum())
    if n_drop:
        reason = "undefined label" if model in NAN_TOLERANT_MODELS else "undefined label or enrichment"
        logger.info(f"  [{y.name}] dropped {n_drop} loops ({reason})")

    return X[keep], y[keep].astype(int)


def build_classifier(model: str, n_jobs: int):
    """Unfitted classifier with the Drosophila time-specific hyperparameters; only the thread count follows the job."""
    params = dict(TIME_SPECIFIC_MODEL_PARAMS[model])
    if "n_jobs" in params:
        params["n_jobs"] = n_jobs
    return MODELS[model]["class"](**params)


def run_condition(enrichment: pd.DataFrame, y: pd.Series, model: str, n_splits: int,
                  n_jobs: int, n_cells: int, n_groups: int) -> tuple[dict, CVResult]:
    """Cross-validate one condition and return its summary row and CVResult.

    The loops are chosen by select_training_loops (see its NaN policy).
    """
    X, y = select_training_loops(enrichment, y, model)

    # Cross-validate with the shared Drosophila harness and hyperparameters.
    classifier = build_classifier(model, n_jobs)
    result = cross_validate(classifier, X, y, KFold(n_splits=n_splits, shuffle=True, random_state=0), None)

    logger.info(f"  {model} | {y.name}: ROC-AUC={result.mean_auc:.4f} +/- {result.std_auc:.4f} | "
                f"ACC={result.mean_acc:.4f} | loops={len(y)} pos={int(y.sum())}")

    row = {
        "condition": y.name, "model": model, "n_cells": n_cells, "n_groups": n_groups,
        "n_features": X.shape[1],
        "n_loops": len(y), "positives": int(y.sum()),
        "mean_auc": round(result.mean_auc, 6), "std_auc": round(result.std_auc, 6),
        "mean_acc": round(result.mean_acc, 6),
    }
    return row, result


def main() -> None:
    parser = argparse.ArgumentParser(description="Human Stage 1: one condition's XGBoost loop-presence classifier.")
    parser.add_argument("--stage0_dir", type=Path, default=Path("results/human_prototype/prepare_data"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/human_prototype/train"))
    parser.add_argument("--feature_set", default=None,
                        help="Feature set name for plot titles; defaults to output_dir's name.")
    parser.add_argument("--model", default="XGB", choices=list(MODELS.keys()))
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument("--n_splits", type=int, default=10)
    parser.add_argument("--n_jobs", type=int, default=12, help="Classifier threads (for models that take n_jobs).")
    parser.add_argument("--log_path", type=Path, default=Path("logs/human_prototype/stage1.log"))
    args = parser.parse_args()

    configure_logging(args.log_path)
    feature_set = args.feature_set or args.output_dir.name

    enrichment = pd.read_csv(args.stage0_dir / "motif_enrichment.csv", index_col=0)
    cells_per_group = pd.read_csv(args.stage0_dir / "n_cells.tsv", sep="\t", index_col=0)["n_cells"]
    n_cells = int(cells_per_group.sum())
    features = describe_features(enrichment.shape[1], len(cells_per_group))
    logger.info(f"Enrichment matrix: {enrichment.shape} (feature set: {feature_set}, {features}, {n_cells} cells)")

    fig_dir = args.output_dir / "figures"
    full = MODELS[args.model]["full"]
    cond = args.condition

    y = pd.read_csv(args.stage0_dir / f"y_{cond}.csv", index_col=0).iloc[:, 0]
    y.name = cond
    row, result = run_condition(enrichment, y, args.model, args.n_splits, args.n_jobs,
                                n_cells, len(cells_per_group))

    save_cv_result(result, str(args.output_dir / "roc_results" / f"{cond}.pkl"))
    plot_roc(result,
             title=condition_title(full, cond, feature_set, row["n_loops"], features,
                                   n_cells, result.mean_auc, result.std_auc),
             out_paths=[fig_dir / f"roc_{cond}.{fmt}" for fmt in ("png", "pdf")])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(args.output_dir / f"cv_{cond}.csv", index=False)
    logger.info(f"Wrote {args.output_dir / f'cv_{cond}.csv'} and {fig_dir / f'roc_{cond}.png'}")


if __name__ == "__main__":
    main()
