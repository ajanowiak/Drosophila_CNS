# human_prepare_data/run_logit_human.py

"""
Fit a statsmodels Logit on the full feature set for one condition.

The human analogue of the Drosophila first logit model, without SHAP
preselection: every feature of the feature set enters the model. Loops with
NaN enrichment are dropped, since statsmodels cannot take NaN. A numerical
failure of the fit (e.g. a singular matrix) is not caught, so it shows up in
the job's log. Snakemake runs one job per negative mode x feature set x condition.

Inputs (results/human_prototype/<negatives>/prepare_data/<feature_set>/):
  - motif_enrichment.csv, y_<condition>.csv
Outputs (results/human_prototype/<negatives>/logit/<feature_set>/):
  - summary_<condition>.csv (coefficients with BH-adjusted p-values),
    figures/<condition>_coefficients.{png,pdf}, figures/<condition>_volcano.{png,pdf}
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from core.log import configure_logging
from core.logit_analysis import fit_logit_summary
from core.logit_plots import plot_coeffs, plot_volcano
from human_prepare_data.constants import CONDITIONS

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a statsmodels Logit on the full feature set for one condition.")
    parser.add_argument("--stage0_dir", type=Path, required=True,
                        help="results/human_prototype/<negatives>/prepare_data/<feature_set>")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="results/human_prototype/<negatives>/logit/<feature_set>")
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument("--p_value_threshold", type=float, required=True)
    parser.add_argument("--top_n_coeffs", type=int, required=True)
    parser.add_argument("--volcano_effect_threshold", type=float, required=True)
    parser.add_argument("--log_path", type=Path, required=True)
    args = parser.parse_args()

    configure_logging(args.log_path)
    cond = args.condition

    enrichment = pd.read_csv(args.stage0_dir / "motif_enrichment.csv", index_col=0)
    y = pd.read_csv(args.stage0_dir / f"y_{cond}.csv", index_col=0).iloc[:, 0]

    # statsmodels cannot take NaN, so drop loops with undefined enrichment
    X = enrichment.loc[y.index]
    keep = X.notna().all(axis=1)
    X, y = X[keep], y[keep]
    logger.info(f"[{cond}] dropped {int((~keep).sum())} loops with NaN enrichment; "
                f"fitting {len(y)} loops ({int(y.sum())} positives) x {X.shape[1]} features")

    summary_df = fit_logit_summary(X, y)
    n_significant = int((summary_df.drop(index="const")["p_adjusted_bh"] < args.p_value_threshold).sum())
    logger.info(f"[{cond}] {n_significant} / {X.shape[1]} features with BH-adjusted p < {args.p_value_threshold}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.output_dir / f"summary_{cond}.csv")

    # the intercept stays in the model and the table, but is not a feature to plot
    plot_df = summary_df.drop(index="const")
    fig_dir = args.output_dir / "figures"
    plot_coeffs(plot_df,
                title=f"Top {args.top_n_coeffs} features with 95% CI - {cond}",
                out_paths=[fig_dir / f"{cond}_coefficients.{fmt}" for fmt in ("png", "pdf")],
                top_n=args.top_n_coeffs)
    plot_volcano(plot_df,
                 title=f"Volcano plot - {cond} ({X.shape[1]} features)\nP-value threshold: {args.p_value_threshold}",
                 out_paths=[fig_dir / f"{cond}_volcano.{fmt}" for fmt in ("png", "pdf")],
                 p_thresh=args.p_value_threshold,
                 effect_thresh=args.volcano_effect_threshold)
    logger.info(f"Wrote {args.output_dir / f'summary_{cond}.csv'} and figures")


if __name__ == "__main__":
    main()
