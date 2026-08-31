# logit_regression_aucroc.py

"""
Cross-validates the first logit model across multiple EPV values, to help
choose an EPV for one tissue.

Pipeline context: compares the same shap_model's downsampled features at
several EPV budgets, reusing regression_coefs.py's own output to also
report how many of those features turned out significant after the
Benjamini-Hochberg correction. Looping over epv values here is intentional:
comparing EPV choices for one tissue is this script's whole purpose,
unlike looping over tissues or models, which Snakemake fans out over.

Inputs:
  - results/shap/<shap_model>/<tissue>_shap_table_<shap_model>.csv
  - results/training_data/unfiltered/hrs<window>/motif_enrichment_hrs<window>.csv
  - results/training_data/unfiltered/hrs<window>/y_<tissue>.csv
  - results/regression_coefs/<shap_model>/epv_<epv>/<tissue>_summary.csv (per epv)

Outputs:
  - results/logit_regression_cv_aucroc/<shap_model>/<tissue>_logit_cv_results.csv
"""

import argparse
import logging
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from core.constants import MODELS, FeatureMode
from core.features import stack_windows
from core.log import configure_logging
from core.logit_analysis import logit_cross_validate
from first_logit_model.features import num_features_from_epv, downsample_features_shap

logger = logging.getLogger(__name__)


def train_logit_cv(tissue: str, shap_model: str, epv: int, n_splits: int, p_value_threshold: float) -> dict:
    """
    Cross-validate the first logit model for one (tissue, shap_model, epv).

    Returns a dict with the CV AUC and the BH-significant feature count
    from regression_coefs.py's own full-data fit for the same combination.
    """
    num_features = num_features_from_epv(tissue, epv)
    logger.info(f"[{tissue}] shap_model={shap_model}, EPV={epv}, num_features={num_features}")

    downsampled_features = downsample_features_shap(tissue, shap_model, num_features)
    X, y, composite = stack_windows(tissue, feature_mode=FeatureMode.CURRENT)
    X = X[downsampled_features]

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    mean_auc, std_auc = logit_cross_validate(X, y, splitter, stratify_target=composite)

    summary_path = f"results/regression_coefs/{shap_model}/epv_{epv}/{tissue}_summary.csv"
    regression_summary = pd.read_csv(summary_path)

    num_significant_bh = (regression_summary["p_adjusted_bh"] < p_value_threshold).sum()

    logger.info(
        f"[{tissue}] AUC={mean_auc:.4f} ± {std_auc:.4f}, "
        f"significant_features={num_significant_bh} (Benjamini-Hochberg)"
    )

    return {
        "tissue": tissue,
        "epv": epv,
        "num_features": num_features,
        "num_features_significant_bh": num_significant_bh,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-validate the first logit model across EPV values for one tissue"
    )
    parser.add_argument("--tissue", required=True, help="Tissue name, e.g. Glia")
    parser.add_argument("--shap_model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--epv_values", type=int, nargs="+", required=True)
    parser.add_argument("--n_splits", type=int, required=True)
    parser.add_argument("--p_value_threshold", type=float, required=True)
    parser.add_argument("--log_path", type=Path, required=True)
    args = parser.parse_args()

    configure_logging(args.log_path)

    all_results = []
    for epv in args.epv_values:
        try:
            all_results.append(
                train_logit_cv(args.tissue, args.shap_model, epv, args.n_splits, args.p_value_threshold)
            )
        except Exception as e:
            logger.info(f"[{args.tissue}] EPV={epv} FAILED: {e}")
            continue

    if not all_results:
        logger.info(f"No successful models for tissue: {args.tissue}")
        return

    results_df = pd.DataFrame(all_results)
    results_df["mean_auc"] = results_df["mean_auc"].round(6)
    results_df["std_auc"] = results_df["std_auc"].round(6)
    results_df = results_df.sort_values("mean_auc", ascending=False)
    results_df = results_df[["epv", "num_features", "num_features_significant_bh", "mean_auc", "std_auc"]]

    output_dir = f"results/logit_regression_cv_aucroc/{args.shap_model}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{args.tissue}_logit_cv_results.csv"
    results_df.to_csv(output_path, index=False)
    logger.info(f"Table for tissue {args.tissue} saved at {output_path}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
