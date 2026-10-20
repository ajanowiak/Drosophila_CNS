# human_prepare_data/summarize_stage1.py

"""
Collect the per-condition Stage 1 results of one (negative mode, feature set):
the CV summary table and the combined ROC figure overlaying every condition.

Stage 1 runs as one Snakemake job per condition (run_stage1_human.py); this
step only reads their outputs, so no classifier is trained here.

Inputs (results/human_prototype/<negatives>/train/<feature_set>/):
  - cv_<condition>.csv, roc_results/<condition>.pkl
Outputs (same directory):
  - cv_summary.csv, figures/roc_combined.{png,pdf}
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from core.constants import MODELS
from core.log import configure_logging
from human_prepare_data.constants import CONDITIONS
from human_prepare_data.run_stage1_human import describe_features
from train_full_models.cv import CVResult
from train_full_models.io import load_cv_result

logger = logging.getLogger(__name__)


def plot_combined_roc(results: dict[str, CVResult], summary: pd.DataFrame,
                      feature_set: str, out_dir: Path) -> None:
    """Overlay every condition's ROC curve into one figure.

    Each condition has its own loop set, so the per-condition loop count goes
    in the legend rather than as a single figure-wide count in the title.
    """
    first = summary.iloc[0]
    full = MODELS[first["model"]]["full"]
    features = describe_features(int(first["n_features"]), int(first["n_groups"]))
    n_loops_by_cond = summary.set_index("condition")["n_loops"]

    fig, ax = plt.subplots(figsize=(6, 6))
    for cond, result in results.items():
        ax.plot(result.mean_fpr, result.mean_tpr,
                label=f"{cond} (AUC = {result.mean_auc:.3f}, {n_loops_by_cond[cond]:,} loops)")
        ax.fill_between(result.mean_fpr, result.tprs_lower, result.tprs_upper, alpha=0.2)

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.grid(axis="both")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title=f"{full} ROC by loop label, {feature_set} features\n"
                 f"{features}, {int(first['n_cells']):,} cells")
    ax.legend(loc="lower right")

    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ("png", "pdf"):
        fig.savefig(out_dir / f"roc_combined.{fmt}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect one feature set's per-condition Stage 1 results.")
    parser.add_argument("--train_dir", type=Path, required=True,
                        help="results/human_prototype/<negatives>/train/<feature_set>")
    parser.add_argument("--feature_set", default=None,
                        help="Feature set name for the plot title; defaults to train_dir's name.")
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS)
    parser.add_argument("--log_path", type=Path, default=Path("logs/human_prototype/summarize_stage1.log"))
    args = parser.parse_args()

    configure_logging(args.log_path)
    feature_set = args.feature_set or args.train_dir.name

    summary = pd.concat([pd.read_csv(args.train_dir / f"cv_{cond}.csv") for cond in args.conditions],
                        ignore_index=True)
    results = {cond: load_cv_result(str(args.train_dir / "roc_results" / f"{cond}.pkl"))
               for cond in args.conditions}

    summary.to_csv(args.train_dir / "cv_summary.csv", index=False)
    logger.info(f"CV summary:\n{summary.to_string(index=False)}")
    plot_combined_roc(results, summary, feature_set, args.train_dir / "figures")
    logger.info(f"Wrote {args.train_dir / 'cv_summary.csv'} and the combined ROC figure")


if __name__ == "__main__":
    main()
