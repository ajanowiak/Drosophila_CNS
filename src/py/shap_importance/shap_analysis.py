# shap_analysis.py

"""
Computes SHAP importance for a trained time-agnostic classifier and saves a
beeswarm plot plus a per-motif importance table.

Pipeline context: Snakemake fans out over model and tissue. Reads the
all-data time-agnostic model trained by train_full_models (curr feature
mode) and rebuilds the same stacked dataset it was trained on, via
core.features.stack_windows. The saved SHAP table feeds the feature preselection for future models.

Inputs:
  - results/models/time_agnostic/<model>/curr/<model>_<tissue>_curr.pkl
  - results/training_data/unfiltered/hrs<window>/motif_enrichment_hrs<window>.csv
  - results/training_data/unfiltered/hrs<window>/y_<tissue>.csv
  - data/motif_names.tsv (motif ID -> name annotations)

Outputs:
  - results/figures/shap/<model>/<tissue>_beeswarm_<model>.pdf
  - results/shap/<model>/<tissue>_shap_table_<model>.csv
"""

import argparse
import logging
from pathlib import Path

from core.constants import MODELS, FeatureMode
from core.features import stack_windows
from core.log import configure_logging
from core.motif_labels import load_motif_annotations, motif_display_labels
from shap_importance.explain import compute_shap_values, summarize_shap_values
from shap_importance.io import load_model, save_shap_table
from shap_importance.plotting import plot_beeswarm

logger = logging.getLogger(__name__)


def run_shap_analysis(
    model: str, tissue: str, motif_annotations_path: str, motif_annotations_sep: str
) -> None:
    """Compute and save SHAP importance for one (model, tissue)."""
    full = MODELS[model]["full"]

    logger.info(f"SHAP analysis: {full} | {tissue}")

    classifier_path = f"results/models/time_agnostic/{model}/curr/{model}_{tissue}_curr.pkl"
    classifier = load_model(classifier_path)

    X, _, _ = stack_windows(tissue, feature_mode=FeatureMode.CURRENT)

    id_to_name = load_motif_annotations(motif_annotations_path, motif_annotations_sep)
    feature_names = motif_display_labels(X.columns, id_to_name)
    n_unmatched = sum(1 for name, motif_id in zip(feature_names, X.columns) if name == motif_id)
    if n_unmatched > 0:
        logger.info(f"{n_unmatched} motif columns had no matching annotation")

    shap_values = compute_shap_values(classifier, model, X)

    beeswarm_path = f"results/figures/shap/{model}/{tissue}_beeswarm_{model}.pdf"
    plot_beeswarm(
        shap_values, X, feature_names,
        title=f"{full} beeswarm SHAP feature importance plot for tissue {tissue}",
        out_path=beeswarm_path,
    )
    logger.info(f"Beeswarm plot saved to {beeswarm_path}")

    shap_df = summarize_shap_values(shap_values, X, id_to_name)
    table_path = f"results/shap/{model}/{tissue}_shap_table_{model}.csv"
    save_shap_table(shap_df, table_path)
    logger.info(f"SHAP table saved to {table_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SHAP importance analysis for a trained time-agnostic classifier"
    )
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--tissue", required=True, help="Tissue name, e.g. Glia")
    parser.add_argument("--motif_annotations_path", default="data/motif_names.tsv")
    parser.add_argument("--motif_annotations_sep", default="\t")
    parser.add_argument("--log_path", type=Path, required=True)
    args = parser.parse_args()

    configure_logging(args.log_path)
    run_shap_analysis(args.model, args.tissue, args.motif_annotations_path, args.motif_annotations_sep)
    logger.info("Done.")


if __name__ == "__main__":
    main()
