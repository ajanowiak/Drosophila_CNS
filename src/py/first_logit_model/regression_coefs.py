# regression_coefs.py

"""
Fits a statsmodels Logit on SHAP-ranked, EPV-downsampled motif features --
the first logit model.

Pipeline context: Snakemake fans out over shap_model, tissue, and epv.
Preselects num_features_from_epv(tissue, epv) motifs, ranked by the SHAP
importance table computed by shap_importance for shap_model, then fits a
single statsmodels Logit on the full dataset and saves its coefficient
summary, a coefficient forest plot, and a volcano plot. This preselection
avoids numerical problems from collinear features and enables extraction
of coefficient statistics for the next stage's feature selection.

Inputs:
  - results/shap/<shap_model>/<tissue>_shap_table_<shap_model>.csv
  - results/training_data/unfiltered/hrs<window>/motif_enrichment_hrs<window>.csv
  - results/training_data/unfiltered/hrs<window>/y_<tissue>.csv
  - data/motif_names.tsv (motif ID -> name annotations)

Outputs:
  - results/regression_coefs/<shap_model>/epv_<epv>/<tissue>_summary.csv
  - results/figures/regression_coefs/<shap_model>/epv_<epv>/<tissue>_coefficients.pdf
  - results/figures/regression_coefs/<shap_model>/epv_<epv>/<tissue>_volcano.pdf
"""

import argparse
import logging
from pathlib import Path

from core.constants import MODELS, FeatureMode
from core.features import stack_windows
from core.log import configure_logging
from core.logit_analysis import fit_logit_summary
from core.logit_plots import plot_coeffs, plot_volcano
from core.motif_labels import load_motif_annotations, motif_display_labels
from first_logit_model.features import num_features_from_epv, downsample_features_shap
from first_logit_model.io import save_summary_table

logger = logging.getLogger(__name__)


def run_regression_coefs(
    tissue: str,
    shap_model: str,
    epv: int,
    motif_annotations_path: str,
    motif_annotations_sep: str,
    top_n_coeffs: int,
    p_value_threshold: float,
    volcano_effect_threshold: float,
) -> None:
    """Fit and save the first logit model for one (tissue, shap_model, epv)."""
    logger.info(f"Regression coefficients: {tissue} | shap_model={shap_model} | epv={epv}")

    num_features = num_features_from_epv(tissue, epv)
    logger.info(f"Selected {num_features} features for tissue {tissue} at EPV={epv}")

    downsampled_features = downsample_features_shap(tissue, shap_model, num_features)

    X, y, _ = stack_windows(tissue, feature_mode=FeatureMode.CURRENT)
    X = X[downsampled_features]

    summary_df = fit_logit_summary(X, y)

    id_to_name = load_motif_annotations(motif_annotations_path, motif_annotations_sep)
    summary_df["motif_name"] = [id_to_name.get(motif_id, motif_id) for motif_id in summary_df.index]

    data_dir = f"results/regression_coefs/{shap_model}/epv_{epv}"
    fig_dir = f"results/figures/regression_coefs/{shap_model}/epv_{epv}"

    summary_path = f"{data_dir}/{tissue}_summary.csv"
    save_summary_table(summary_df, summary_path)
    logger.info(f"Summary table saved to {summary_path}")

    # plot with "name  -  (id)" display labels; the saved table keeps raw motif_id
    plot_df = summary_df.copy()
    plot_df.index = motif_display_labels(summary_df.index, id_to_name)

    coef_plot_path = f"{fig_dir}/{tissue}_coefficients.pdf"
    plot_coeffs(
        plot_df,
        title=f"Top {top_n_coeffs} Features with 95% CI - {tissue}",
        output_path=coef_plot_path,
        top_n=top_n_coeffs,
    )
    logger.info(f"Coefficient plot saved to {coef_plot_path}")

    volcano_plot_path = f"{fig_dir}/{tissue}_volcano.pdf"
    plot_volcano(
        plot_df,
        title=f"Volcano Plot - {tissue} ({num_features} features)\nP-value threshold: {p_value_threshold}",
        output_path=volcano_plot_path,
        p_thresh=p_value_threshold,
        effect_thresh=volcano_effect_threshold,
    )
    logger.info(f"Volcano plot saved to {volcano_plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit the first logit model on SHAP-ranked, EPV-downsampled features"
    )
    parser.add_argument("--tissue", required=True, help="Tissue name, e.g. Glia")
    parser.add_argument("--shap_model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--epv", type=int, required=True)
    parser.add_argument("--motif_annotations_path", default="data/motif_names.tsv")
    parser.add_argument("--motif_annotations_sep", default="\t")
    parser.add_argument("--top_n_coeffs", type=int, required=True)
    parser.add_argument("--p_value_threshold", type=float, required=True)
    parser.add_argument("--volcano_effect_threshold", type=float, required=True)
    parser.add_argument("--log_path", type=Path, required=True)
    args = parser.parse_args()

    configure_logging(args.log_path)
    run_regression_coefs(
        args.tissue, args.shap_model, args.epv,
        args.motif_annotations_path, args.motif_annotations_sep,
        args.top_n_coeffs, args.p_value_threshold, args.volcano_effect_threshold,
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
