# logit_regression_significant_features.py

"""
Retrains the logit model using only the statistically significant motifs
from the first logit model - the second logit model.

Pipeline context: Snakemake fans out over shap_model, tissue, and
feature_selection_mode (plus epv, when that mode is epv) - the same
selection used to produce the regression_coefs.py output this stage reads.
Retaining only BH-significant motifs and refitting produces a small,
interpretable model whose predictive performance supports the biological
relevance of the selected motifs.

Inputs:
  - results/first_logit_model/regression_coefs/tables/<shap_model>/<selection_tag>/<tissue>.csv
  - results/prepare_data/unfiltered/hrs<window>/motif_enrichment.csv
  - results/prepare_data/unfiltered/hrs<window>/y_<tissue>.csv
  - data/motif_names.tsv (motif ID -> name annotations)

Outputs:
  - results/second_logit_model/logit_regression_significant_features/tables/<shap_model>/<selection_tag>/<tissue>_summary.csv
  - results/second_logit_model/logit_regression_significant_features/tables/<shap_model>/<selection_tag>/<tissue>_cv_results.csv
  - results/second_logit_model/logit_regression_significant_features/figures/<shap_model>/<selection_tag>/<tissue>_coefficients.{pdf,png}
  - results/second_logit_model/logit_regression_significant_features/figures/<shap_model>/<selection_tag>/<tissue>_volcano.{pdf,png}
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from core.constants import MODELS, FeatureMode, FeatureSelectionMode
from core.features import stack_windows
from core.log import configure_logging
from core.logit_analysis import fit_logit_summary, logit_cross_validate
from core.logit_plots import plot_coeffs, plot_volcano, plot_failed_placeholder
from core.motif_labels import load_motif_annotations, motif_display_labels
from core.paths import second_logit_model_figure_dir, second_logit_model_table_dir
from first_logit_model.features import feature_selection_tag
from second_logit_model.io import extract_significant_features, save_summary_table, save_cv_result_row

logger = logging.getLogger(__name__)


def run_significant_features(
    tissue: str,
    shap_model: str,
    selection_tag: str,
    n_splits: int,
    p_value_threshold: float,
    top_n_coeffs: int,
    volcano_effect_threshold: float,
    motif_annotations_path: str,
    motif_annotations_sep: str,
) -> None:
    """Fit and save the second logit model for one (tissue, shap_model, selection_tag)."""
    logger.info(f"Significant features: {tissue} | shap_model={shap_model} | selection={selection_tag}")

    data_dir = second_logit_model_table_dir(shap_model, selection_tag)
    fig_dir = second_logit_model_figure_dir(shap_model, selection_tag)
    cv_results_path = f"{data_dir}/{tissue}_cv_results.csv"
    summary_path = f"{data_dir}/{tissue}_summary.csv"
    coef_plot_paths = [f"{fig_dir}/{tissue}_coefficients.pdf", f"{fig_dir}/{tissue}_coefficients.png"]
    volcano_plot_paths = [f"{fig_dir}/{tissue}_volcano.pdf", f"{fig_dir}/{tissue}_volcano.png"]
    empty_summary_columns = [
        "coef", "std_err", "z", "p_unadjusted", "p_adjusted_bh",
        "p_adjusted_tsbh", "ci_lower", "ci_upper", "motif_name",
    ]

    significant_features = extract_significant_features(tissue, shap_model, selection_tag, p_value_threshold)

    if len(significant_features) == 0:
        # a legitimate, expected outcome (not every selection has any
        # BH-significant motif) - still write the declared outputs so this
        # is a real, inspectable "nothing significant" result rather than a
        # missing file breaking the Snakemake DAG.
        logger.info(f"[{tissue}] No significant features found at p < {p_value_threshold}")
        save_cv_result_row(
            {"tissue": tissue, "num_features": 0, "mean_auc": None, "std_auc": None},
            cv_results_path,
        )
        save_summary_table(pd.DataFrame(columns=empty_summary_columns), summary_path)
        message = f"No significant features for {tissue} ({shap_model}, {selection_tag})"
        plot_failed_placeholder(message, coef_plot_paths)
        plot_failed_placeholder(message, volcano_plot_paths)
        return

    logger.info(f"[{tissue}] Found {len(significant_features)} significant features")

    X, y, composite = stack_windows(tissue, feature_mode=FeatureMode.CURRENT)
    X = X[significant_features]

    # a small significant-feature subset is much less prone to this than
    # Stage 3's large SHAP-ranked prefixes, but a fold (CV) or the full-data
    # refit can still hit a singular design matrix.
    try:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        mean_auc, std_auc = logit_cross_validate(X, y, splitter, stratify_target=composite)
        logger.info(f"[{tissue}] AUC={mean_auc:.4f} ± {std_auc:.4f} ({len(significant_features)} significant features)")

        summary_df = fit_logit_summary(X, y)
    except (np.linalg.LinAlgError, PerfectSeparationError) as e:
        logger.info(f"[{tissue}] shap_model={shap_model} selection={selection_tag} FIT FAILED: {e}")
        save_cv_result_row(
            {"tissue": tissue, "num_features": len(significant_features), "mean_auc": None, "std_auc": None},
            cv_results_path,
        )
        save_summary_table(pd.DataFrame(columns=empty_summary_columns), summary_path)
        message = f"Fit failed for {tissue} ({len(significant_features)} features, {shap_model}, {selection_tag}):\n{e}"
        plot_failed_placeholder(message, coef_plot_paths)
        plot_failed_placeholder(message, volcano_plot_paths)
        return

    id_to_name = load_motif_annotations(motif_annotations_path, motif_annotations_sep)
    summary_df["motif_name"] = [id_to_name.get(motif_id, motif_id) for motif_id in summary_df.index]

    save_cv_result_row(
        {
            "tissue": tissue,
            "num_features": len(significant_features),
            "mean_auc": mean_auc,
            "std_auc": std_auc,
        },
        cv_results_path,
    )
    logger.info(f"Saved CV results to {cv_results_path}")

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
        title=f"Volcano Plot - {tissue} ({len(significant_features)} features)\nP-value threshold: {p_value_threshold}",
        out_paths=volcano_plot_paths,
        p_thresh=p_value_threshold,
        effect_thresh=volcano_effect_threshold,
    )
    logger.info(f"Volcano plot saved to {volcano_plot_paths}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrain the logit model using only significant motifs from the first logit model"
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
    parser.add_argument("--n_splits", type=int, required=True)
    parser.add_argument("--p_value_threshold", type=float, required=True)
    parser.add_argument("--top_n_coeffs", type=int, required=True)
    parser.add_argument("--volcano_effect_threshold", type=float, required=True)
    parser.add_argument("--motif_annotations_path", default="data/motif_names.tsv")
    parser.add_argument("--motif_annotations_sep", default="\t")
    parser.add_argument("--log_path", type=Path, required=True)
    args = parser.parse_args()

    if args.feature_selection_mode == FeatureSelectionMode.EPV and args.epv is None:
        parser.error("--epv is required when --feature_selection_mode is epv")

    configure_logging(args.log_path)
    selection_tag = feature_selection_tag(args.feature_selection_mode, args.epv)
    run_significant_features(
        args.tissue, args.shap_model, selection_tag, args.n_splits,
        args.p_value_threshold, args.top_n_coeffs, args.volcano_effect_threshold,
        args.motif_annotations_path, args.motif_annotations_sep,
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
