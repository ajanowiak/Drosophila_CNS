# human_prepare_data/run_shap_human.py

"""
Human Stage 2 driver: SHAP importance for one condition's loop-presence classifier.

Mirrors the Drosophila shap_importance/shap_analysis.py and reuses its SHAP
computation, summary table and beeswarm plot unchanged. Human Stage 1 only
cross-validates and keeps no fitted model, so this script fits the all-data
classifier itself - same loops, model, hyperparameters and NaN policy as
run_stage1_human.py - saves it, and explains it with SHAP on those same loops.

Snakemake runs one job per negative mode x feature set x condition. Features
are pooled motifs or motif x cell group ("<motif>__<group>"); the table splits
each feature into its JASPAR matrix id, motif name and cell group, so the
groups of one motif can be compared downstream.

Inputs (results/human_prototype/<negatives>/prepare_data/<feature_set>/):
  - motif_enrichment.csv, y_<condition>.csv, n_cells.tsv
Outputs (results/human_prototype/<negatives>/shap/<feature_set>/):
  - models/<condition>.pkl (classifier fitted on all of the condition's loops)
  - tables/<condition>.csv (per-feature SHAP summary)
  - figures/<condition>.{png,pdf} (beeswarm)
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from core.constants import MODELS
from core.log import configure_logging
from human_prepare_data.constants import CONDITIONS, GROUP_SEPARATOR, POOLED_GROUP
from human_prepare_data.run_stage1_human import build_classifier, select_training_loops
from shap_importance.explain import compute_shap_values, summarize_shap_values
from shap_importance.io import save_shap_table
from shap_importance.plotting import plot_beeswarm
from train_full_models.cv import fit_full_data_model
from train_full_models.io import save_model

logger = logging.getLogger(__name__)


def annotate_features(features: pd.Index, grouped: bool) -> pd.DataFrame:
    """Split each feature name into JASPAR matrix id, motif name and cell group.

    chromVAR names motifs "<matrix id>.<name>" (e.g. MA0004.1.Arnt); grouped
    features append "__<group>", pooled ones get the pooled group name.
    """
    rows = []
    for feature in features:
        motif, group = feature.rsplit(GROUP_SEPARATOR, 1) if grouped else (feature, POOLED_GROUP)
        base_id, version, name = motif.split(".", 2)
        rows.append((f"{base_id}.{version}", name, group))

    return pd.DataFrame(rows, index=features, columns=["matrix_id", "motif_name", "group"])


def display_labels(annotation: pd.DataFrame, grouped: bool) -> list[str]:
    """Beeswarm labels in the Drosophila "name  -  (id)" style, with the cell group when grouped."""
    labels = []
    for matrix_id, name, group in annotation.itertuples(index=False):
        shown = f"{name} [{group}]" if grouped else name
        labels.append(f"{shown}  -  ({matrix_id})")
    return labels


def summarize_features(shap_values, X: pd.DataFrame, annotation: pd.DataFrame) -> pd.DataFrame:
    """Drosophila per-feature SHAP summary, with each feature's matrix id, motif name and group.

    Sorted by absolute mean importance, as in the Drosophila table.
    """
    summary = (summarize_shap_values(shap_values, X, id_to_name={})
               .drop(columns="motif_name")
               .rename(columns={"motif_id": "feature"})
               .join(annotation, on="feature"))
    stat_columns = ["mean_importance", "abs_mean_importance", "mean_abs_importance", "std_importance"]
    return summary[["feature", "matrix_id", "motif_name", "group", *stat_columns]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Human Stage 2: SHAP importance for one condition's classifier.")
    parser.add_argument("--stage0_dir", type=Path, required=True,
                        help="One (negative mode, feature set)'s prepare_data directory.")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="One (negative mode, feature set)'s shap directory.")
    parser.add_argument("--feature_set", default=None,
                        help="Feature set name for the plot title; defaults to output_dir's name.")
    parser.add_argument("--model", default="XGB", choices=list(MODELS.keys()))
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument("--n_jobs", type=int, default=4, help="Classifier threads (for models that take n_jobs).")
    parser.add_argument("--log_path", type=Path, required=True)
    args = parser.parse_args()

    configure_logging(args.log_path)
    feature_set = args.feature_set or args.output_dir.name
    cond = args.condition
    full = MODELS[args.model]["full"]
    logger.info(f"SHAP analysis: {full} | {cond} | feature set {feature_set}")

    enrichment = pd.read_csv(args.stage0_dir / "motif_enrichment.csv", index_col=0)
    groups = pd.read_csv(args.stage0_dir / "n_cells.tsv", sep="\t", index_col=0).index
    grouped = list(groups) != [POOLED_GROUP]
    y = pd.read_csv(args.stage0_dir / f"y_{cond}.csv", index_col=0).iloc[:, 0]
    y.name = cond

    X, y = select_training_loops(enrichment, y, args.model)
    logger.info(f"Training on {len(y)} loops ({int(y.sum())} positives) x {X.shape[1]} features")

    classifier = fit_full_data_model(build_classifier(args.model, args.n_jobs), X, y)
    model_path = args.output_dir / "models" / f"{cond}.pkl"
    save_model(classifier, str(model_path))
    logger.info(f"All-data model saved to {model_path}")

    shap_values = compute_shap_values(classifier, args.model, X)
    annotation = annotate_features(X.columns, grouped)

    beeswarm_paths = [args.output_dir / "figures" / f"{cond}.{fmt}" for fmt in ("pdf", "png")]
    plot_beeswarm(
        shap_values, X, display_labels(annotation, grouped),
        title=f"{full} beeswarm SHAP feature importance plot: {cond} loops, {feature_set} features",
        out_paths=beeswarm_paths,
    )
    logger.info(f"Beeswarm plot saved to {beeswarm_paths}")

    table_path = args.output_dir / "tables" / f"{cond}.csv"
    save_shap_table(summarize_features(shap_values, X, annotation), str(table_path))
    logger.info(f"SHAP table saved to {table_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
