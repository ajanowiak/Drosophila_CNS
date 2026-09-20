# human_prepare_data/aggregate_results.py

"""
Collate the per-cell-set Stage 1 summaries into one table of the nine
(ATAC cell set x loop-presence label) classifiers, and run the loop-count
checkpoint: every cell set must train on the SAME loop set for a given label.

Inputs:
  - results/human_prototype/train/<cell_set>/cv_summary.csv  (one per cell set)
Outputs:
  - results/human_prototype/train/all_cellsets_summary.csv
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from core.log import configure_logging
from human_prepare_data.constants import CELL_SETS

logger = logging.getLogger(__name__)


def check_equal_loop_counts(summary: pd.DataFrame) -> None:
    """Checkpoint: for each loop-presence label, every cell set trained on the
    same number of loops.

    Cell sets differ only in which cells define the both-open "11" state, not in
    which loops exist. With NaN-tolerant XGBoost no loop is dropped, so the
    per-condition loop counts must be identical across cell sets. A mismatch
    means loops were silently dropped somewhere and the classifiers are no
    longer comparable across cell sets.
    """
    for cond, grp in summary.groupby("condition"):
        counts = grp.set_index("cell_set")["n_loops"]
        if counts.nunique() != 1:
            raise ValueError(
                f"Loop-count mismatch across cell sets for label '{cond}': "
                f"{counts.to_dict()} - cell sets are no longer comparable."
            )
    logger.info(f"Checkpoint OK: loop counts equal across cell sets for every label "
                f"({summary['cell_set'].nunique()} cell sets).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate the 9 human classifiers into one table.")
    parser.add_argument("--train_dir", type=Path, default=Path("results/human_prototype/train"))
    parser.add_argument("--cell_sets", nargs="*", default=CELL_SETS)
    parser.add_argument("--output", type=Path, default=Path("results/human_prototype/train/all_cellsets_summary.csv"))
    parser.add_argument("--log_path", type=Path, default=Path("logs/human_prototype/aggregate.log"))
    args = parser.parse_args()

    configure_logging(args.log_path)

    rows = []
    for cell_set in args.cell_sets:
        path = args.train_dir / cell_set / "cv_summary.csv"
        if not path.exists():
            logger.warning(f"missing {path} - skipping")
            continue
        df = pd.read_csv(path)
        df.insert(0, "cell_set", cell_set)
        rows.append(df)

    summary = pd.concat(rows, ignore_index=True)
    summary.to_csv(args.output, index=False)
    logger.info(f"Wrote {args.output} ({len(summary)} classifiers)")

    check_equal_loop_counts(summary)

    # ROC-AUC grid: rows are ATAC cell sets, columns are loop-presence labels.
    pivot = summary.pivot(index="cell_set", columns="condition", values="mean_auc").reindex(args.cell_sets)
    logger.info(f"ROC-AUC grid (rows = ATAC cell set, cols = loop label):\n{pivot.round(3).to_string()}")

    print("\nROC-AUC (rows = ATAC cell set, cols = loop-presence label):")
    print(pivot.round(3).to_string())
    best_cell_set = summary.loc[summary.mean_auc.idxmax(), "cell_set"]
    best_condition = summary.loc[summary.mean_auc.idxmax(), "condition"]
    print(f"\nBest: {best_cell_set} x {best_condition} = {summary.mean_auc.max():.3f}")


if __name__ == "__main__":
    main()
