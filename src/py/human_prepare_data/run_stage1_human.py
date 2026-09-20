# human_prepare_data/run_stage1_human.py

"""
Human Stage 1 driver: condition-specific loop-presence classifier.

Reuses the Drosophila CV harness (cross_validate), ROC plotting (plot_roc) and
the exact time-specific XGBoost hyperparameters. The human glue loads the
enrichment matrix and per-condition presence vector, drops loops with undefined
enrichment (no both-open cell), and reports/plots ROC per condition.

Inputs (results/human_prototype/prepare_data/):
  - motif_enrichment.csv, y_<condition>.csv, n_cells.txt
Outputs (results/human_prototype/train/):
  - cv_summary.csv, roc_results/<condition>.pkl,
    figures/roc_<condition>.{png,pdf}, figures/roc_combined.{png,pdf}
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold

from core.constants import MODELS, TIME_SPECIFIC_MODEL_PARAMS
from core.log import configure_logging
from human_prepare_data.constants import CELL_DISPLAY, CONDITIONS, NAN_TOLERANT_MODELS
from train_full_models.cv import CVResult, cross_validate
from train_full_models.io import save_cv_result
from train_full_models.plotting import plot_roc

logger = logging.getLogger(__name__)


def condition_title(full: str, cond: str, cell_set: str, n_loops: int, n_features: int,
                    n_cells: int, mean_auc: float, std_auc: float) -> str:
    """Drosophila-style three-line ROC title: what/from-what, sizes, then AUC."""
    cells = CELL_DISPLAY.get(cell_set, cell_set)
    title = (
        f"{full} ROC: {cond} loops, {cells} cells\n"
        f"{n_loops:,} loops x {n_features:,} motifs, {n_cells:,} cells\n"
        f"AUC = {mean_auc:.3f} ± {std_auc:.3f}"
    )
    return title


def run_condition(enrichment: pd.DataFrame, y: pd.Series, model: str, n_splits: int,
                  n_cells: int) -> tuple[dict, CVResult]:
    """Cross-validate one condition and return its summary row and CVResult.

    NaN policy: a loop with no both-open "11" cell in this cell set has an
    undefined (NaN) enrichment row. For a NaN-tolerant model (XGBoost) we KEEP
    such loops - the classifier routes the missing values, and every cell set
    then trains on the identical loop set (checked in aggregate_results.py). For
    a model that cannot take NaN we fall back to dropping those rows.
    """
    X = enrichment.loc[y.index]

    # Decide which loops to train on (see the NaN policy above).
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

    X, y = X[keep], y[keep].astype(int)

    # Cross-validate with the shared Drosophila harness and hyperparameters.
    classifier = MODELS[model]["class"](**TIME_SPECIFIC_MODEL_PARAMS[model])
    result = cross_validate(classifier, X, y, KFold(n_splits=n_splits, shuffle=True, random_state=0), None)

    logger.info(f"  {model} | {y.name}: ROC-AUC={result.mean_auc:.4f} +/- {result.std_auc:.4f} | "
                f"ACC={result.mean_acc:.4f} | loops={len(y)} pos={int(y.sum())}")

    row = {
        "condition": y.name, "model": model, "n_cells": n_cells,
        "n_loops": len(y), "positives": int(y.sum()),
        "mean_auc": round(result.mean_auc, 6), "std_auc": round(result.std_auc, 6),
        "mean_acc": round(result.mean_acc, 6),
    }
    return row, result


def plot_combined_roc(results: dict[str, CVResult], model: str, cell_set: str,
                      n_loops: int, n_features: int, n_cells: int, out_dir: Path) -> None:
    """Overlay every condition's ROC curve into one figure."""
    full = MODELS[model]["full"]
    cells = CELL_DISPLAY.get(cell_set, cell_set)

    fig, ax = plt.subplots(figsize=(6, 6))
    for cond, result in results.items():
        ax.plot(result.mean_fpr, result.mean_tpr, label=f"{cond} (AUC = {result.mean_auc:.3f})")
        ax.fill_between(result.mean_fpr, result.tprs_lower, result.tprs_upper, alpha=0.2)

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.grid(axis="both")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title=f"{full} ROC by loop label, {cells} cells\n"
                 f"{n_loops:,} loops x {n_features:,} motifs, {n_cells:,} cells")
    ax.legend(loc="lower right")

    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ("png", "pdf"):
        fig.savefig(out_dir / f"roc_combined.{fmt}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Human Stage 1: condition-specific XGBoost loop-presence classifier.")
    parser.add_argument("--stage0_dir", type=Path, default=Path("results/human_prototype/prepare_data"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/human_prototype/train"))
    parser.add_argument("--cell_set", default=None,
                        help="ATAC cell set for plot titles; defaults to output_dir's name.")
    parser.add_argument("--model", default="XGB", choices=list(MODELS.keys()))
    parser.add_argument("--conditions", nargs="*", default=CONDITIONS)
    parser.add_argument("--n_splits", type=int, default=10)
    parser.add_argument("--log_path", type=Path, default=Path("logs/human_prototype/stage1.log"))
    args = parser.parse_args()

    configure_logging(args.log_path)
    cell_set = args.cell_set or args.output_dir.name

    enrichment = pd.read_csv(args.stage0_dir / "motif_enrichment.csv", index_col=0)
    n_features = enrichment.shape[1]
    n_cells = int((args.stage0_dir / "n_cells.txt").read_text().strip())
    logger.info(f"Enrichment matrix: {enrichment.shape} (cell set: {cell_set}, {n_cells} cells)")

    fig_dir = args.output_dir / "figures"
    full = MODELS[args.model]["full"]

    rows, results = [], {}
    for cond in args.conditions:
        y = pd.read_csv(args.stage0_dir / f"y_{cond}.csv", index_col=0).iloc[:, 0]
        y.name = cond

        row, result = run_condition(enrichment, y, args.model, args.n_splits, n_cells)
        rows.append(row)
        results[cond] = result

        save_cv_result(result, str(args.output_dir / "roc_results" / f"{cond}.pkl"))
        plot_roc(result,
                 title=condition_title(full, cond, cell_set, row["n_loops"], n_features,
                                       n_cells, result.mean_auc, result.std_auc),
                 out_paths=[fig_dir / f"roc_{cond}.{fmt}" for fmt in ("png", "pdf")])

    plot_combined_roc(results, args.model, cell_set, rows[0]["n_loops"], n_features, n_cells, fig_dir)

    summary = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "cv_summary.csv", index=False)
    logger.info(f"Wrote {args.output_dir / 'cv_summary.csv'} and ROC figures to {fig_dir}")
    print(summary.to_string(index=False))

    best = summary.loc[summary["mean_auc"].idxmax()]
    logger.info(f"Best condition: {best['condition']} ROC-AUC={best['mean_auc']:.4f} "
                f"(target >= 0.55: {'PASS' if best['mean_auc'] >= 0.55 else 'FAIL'})")


if __name__ == "__main__":
    main()
