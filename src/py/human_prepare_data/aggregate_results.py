# human_prepare_data/aggregate_results.py

"""
Collate the Stage 1 summaries of every (negative mode x feature set) run into
one table, so the ROC-AUC of e.g. pooled motif features and motif x cell-group
features can be compared under both negative modes, and run the loop-count
checkpoint: within one negative mode, every feature set must train on the SAME
loop set for a given condition.

Inputs:
  - results/human_prototype/<negatives>/train/<feature_set>/cv_summary.csv
Outputs:
  - results/human_prototype/train_summary.csv
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from core.log import configure_logging
from human_prepare_data.constants import NAN_TOLERANT_MODELS

logger = logging.getLogger(__name__)


def check_equal_loop_counts(summary: pd.DataFrame) -> None:
    """Checkpoint: for each (negative mode, condition), every feature set trained on
    the same number of loops.

    A condition's loop set is pure loop geometry, fixed by the negative mode and
    independent of the features, and NaN-tolerant XGBoost drops no loop, so the
    loop count must be identical across feature sets. A mismatch means loops were
    silently dropped somewhere and the feature sets are no longer comparable.
    Models that cannot take NaN drop loops with undefined enrichment, which
    differs between feature sets, so for them a mismatch is only a warning.
    """
    for (negatives, cond), grp in summary.groupby(["negatives", "condition"]):
        counts = grp.set_index("feature_set")["n_loops"]
        if counts.nunique() == 1:
            continue

        message = (f"Loop-count mismatch across feature sets for {negatives} / '{cond}': "
                   f"{counts.to_dict()} - feature sets are not trained on the same loops.")
        if set(grp["model"]) <= NAN_TOLERANT_MODELS:
            raise ValueError(message)
        logger.warning(message)

    logger.info(f"Loop-count checkpoint done ({summary['feature_set'].nunique()} feature sets, "
                f"{summary['negatives'].nunique()} negative modes).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate the human classifiers across negative modes and feature sets.")
    parser.add_argument("--results_dir", type=Path, default=Path("results/human_prototype"))
    parser.add_argument("--negatives", nargs="+", required=True)
    parser.add_argument("--feature_sets", nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=Path("results/human_prototype/train_summary.csv"))
    parser.add_argument("--log_path", type=Path, default=Path("logs/human_prototype/aggregate.log"))
    args = parser.parse_args()

    configure_logging(args.log_path)

    rows = []
    for negatives in args.negatives:
        for feature_set in args.feature_sets:
            path = args.results_dir / negatives / "train" / feature_set / "cv_summary.csv"
            if not path.exists():
                logger.warning(f"missing {path} - skipping")
                continue
            df = pd.read_csv(path)
            df.insert(0, "feature_set", feature_set)
            df.insert(0, "negatives", negatives)
            rows.append(df)

    summary = pd.concat(rows, ignore_index=True)
    summary.to_csv(args.output, index=False)
    logger.info(f"Wrote {args.output} ({len(summary)} classifiers)")

    check_equal_loop_counts(summary)

    # ROC-AUC grid: rows are (negative mode, feature set), columns are loop-presence labels.
    pivot = summary.pivot_table(index=["negatives", "feature_set"], columns="condition", values="mean_auc")
    logger.info(f"ROC-AUC grid (rows = negative mode x feature set, cols = loop label):\n"
                f"{pivot.round(3).to_string()}")

    best = summary.loc[summary["mean_auc"].idxmax()]
    logger.info(f"Best: {best['negatives']} x {best['feature_set']} x {best['condition']} "
                f"= {best['mean_auc']:.3f}")


if __name__ == "__main__":
    main()
