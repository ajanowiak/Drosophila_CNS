# regression_coefs.py

"""
Fits a statsmodels Logit on SHAP-ranked, downsampled motif features - the
first logit model.

Pipeline context: Snakemake fans out over shap_model, tissue, and
feature_selection_mode (plus epv, when that mode is epv). Preselects a
number of motifs, ranked by the SHAP importance table computed by
shap_importance for shap_model, then fits a single statsmodels Logit on the
full dataset and saves its coefficient summary, a coefficient plot,
and a volcano plot. This preselection avoids numerical problems from
collinear features and enables extraction of coefficient statistics for the
next stage's feature selection.

feature_selection_mode picks how many motifs to preselect:
  binsearch - the largest SHAP-ranked prefix that fits without a singular
              matrix error (default)

    epv       - a fixed events-per-variable budget (Number of features 
                computed dynamically with num_features_from_epv)

Inputs:
  - results/shap_importance/shap_analysis/tables/<shap_model>/<tissue>.csv
  - results/prepare_data/unfiltered/hrs<window>/motif_enrichment.csv
  - results/prepare_data/unfiltered/hrs<window>/y_<tissue>.csv
  - data/motif_names.tsv (motif ID -> name annotations)

Outputs:
  - results/first_logit_model/regression_coefs/tables/<shap_model>/<selection_tag>/<tissue>.csv
  - results/first_logit_model/regression_coefs/figures/<shap_model>/<selection_tag>/<tissue>_coefficients.{pdf,png}
  - results/first_logit_model/regression_coefs/figures/<shap_model>/<selection_tag>/<tissue>_volcano.{pdf,png}
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from core.constants import MODELS, FeatureMode, FeatureSelectionMode
from core.features import stack_windows
from core.log import configure_logging
from core.logit_analysis import fit_logit_summary, find_max_features_binary_search
from core.logit_plots import plot_coeffs, plot_volcano, plot_failed_placeholder
from core.motif_labels import load_motif_annotations, motif_display_labels
from core.paths import regression_coefs_figure_dir, regression_coefs_table_path
from first_logit_model.features import (
    num_features_from_epv,
    downsample_features_shap,
    feature_selection_tag,
)
from first_logit_model.io import save_summary_table

logger = logging.getLogger(__name__)


def run_regression_coefs(
    tissue: str,
    shap_model: str,
    feature_selection_mode: FeatureSelectionMode,
    epv: int | None,
    motif_annotations_path: str,
    motif_annotations_sep: str,
    top_n_coeffs: int,
    p_value_threshold: float,
    volcano_effect_threshold: float,
) -> None:
    """Fit and save the first logit model for one (tissue, shap_model, feature_selection_mode)."""

    selection_tag = feature_selection_tag(feature_selection_mode, epv)
    logger.info(f"Regression coefficients: {tissue} | shap_model={shap_model} | selection={selection_tag}")

    X, y, _ = stack_windows(tissue, feature_mode=FeatureMode.CURRENT)
    ranked_features = downsample_features_shap(tissue, shap_model, num_features=None)

    if feature_selection_mode == FeatureSelectionMode.EPV:
        num_features = num_features_from_epv(tissue, epv)
        logger.info(f"Selected {num_features} features for tissue {tissue} at EPV={epv}")
    else:
        num_features = find_max_features_binary_search(X[ranked_features], y)
        logger.info(f"Binary search selected {num_features} features for tissue {tissue}")

    downsampled_features = ranked_features[:num_features]
    X = X[downsampled_features]

    fig_dir = regression_coefs_figure_dir(shap_model, selection_tag)
    summary_path = regression_coefs_table_path(shap_model, selection_tag, tissue)
    coef_plot_paths = [f"{fig_dir}/{tissue}_coefficients.pdf", f"{fig_dir}/{tissue}_coefficients.png"]
    volcano_plot_paths = [f"{fig_dir}/{tissue}_volcano.pdf", f"{fig_dir}/{tissue}_volcano.png"]

    # some epv budgets pick a feature count the full-data fit can't actually
    # support (no feasibility check the way binary search's own search loop
    # has) - expected to happen for some (tissue, epv) combinations, so log
    # and move on rather than crashing the whole run.
    try:
        summary_df = fit_logit_summary(X, y)
    except (np.linalg.LinAlgError, PerfectSeparationError) as e:
        logger.info(f"[{tissue}] shap_model={shap_model} selection={selection_tag} FIT FAILED: {e}")
        empty_columns = [
            "coef", "std_err", "z", "p_unadjusted", "p_adjusted_bh",
            "p_adjusted_tsbh", "ci_lower", "ci_upper", "motif_name",
        ]
        save_summary_table(pd.DataFrame(columns=empty_columns), summary_path)
        message = f"Fit failed for {tissue} ({num_features} features, {selection_tag}):\n{e}"
        plot_failed_placeholder(message, coef_plot_paths)
        plot_failed_placeholder(message, volcano_plot_paths)
        return

    id_to_name = load_motif_annotations(motif_annotations_path, motif_annotations_sep)
    summary_df["motif_name"] = [id_to_name.get(motif_id, motif_id) for motif_id in summary_df.index]

    save_summary_table(summary_df, summary_path)
    logger.info(f"Summary table saved to {summary_path}")

    # plot with "name  -  (id)" display labels; the saved table keeps raw motif_id
    plot_df = summary_df.copy()
    plot_df.index = motif_display_labels(summary_df.index, id_to_name)

    plot_coeffs(
        plot_df,
        title=f"Top {top_n_coeffs} Features with 95% CI - {tissue}",
        out_paths=coef_plot_paths,
        top_n=top_n_coeffs,
    )
    logger.info(f"Coefficient plot saved to {coef_plot_paths}")

    plot_volcano(
        plot_df,
        title=f"Volcano Plot - {tissue} ({num_features} features)\nP-value threshold: {p_value_threshold}",
        out_paths=volcano_plot_paths,
        p_thresh=p_value_threshold,
        effect_thresh=volcano_effect_threshold,
    )
    logger.info(f"Volcano plot saved to {volcano_plot_paths}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit the first logit model on SHAP-ranked, downsampled features"
    )
    parser.add_argument("--tissue", required=True, help="Tissue name, e.g. Glia")
    parser.add_argument("--shap_model", required=True, choices=list(MODELS.keys()))
    parser.add_argument(
        "--feature_selection_mode",
        type=FeatureSelectionMode,
        choices=list(FeatureSelectionMode),
        required=True,
    )
    parser.add_argument(
        "--epv", type=int, default=None,
        help="Required when --feature_selection_mode is epv; ignored otherwise",
    )
    parser.add_argument("--motif_annotations_path", default="data/motif_names.tsv")
    parser.add_argument("--motif_annotations_sep", default="\t")
    parser.add_argument("--top_n_coeffs", type=int, required=True)
    parser.add_argument("--p_value_threshold", type=float, required=True)
    parser.add_argument("--volcano_effect_threshold", type=float, required=True)
    parser.add_argument("--log_path", type=Path, required=True)
    args = parser.parse_args()

    if args.feature_selection_mode == FeatureSelectionMode.EPV and args.epv is None:
        parser.error("--epv is required when --feature_selection_mode is epv")

    configure_logging(args.log_path)
    run_regression_coefs(
        args.tissue, args.shap_model, args.feature_selection_mode, args.epv,
        args.motif_annotations_path, args.motif_annotations_sep,
        args.top_n_coeffs, args.p_value_threshold, args.volcano_effect_threshold,
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
