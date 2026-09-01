# logit_regression_aucroc.py

"""
Cross-validates the first logit model's already-chosen feature set(s), to
help evaluate a tissue's feature selection.

Pipeline context: reads the feature set straight from regression_coefs.py's
saved summary instead of recomputing it - so this never repeats an
expensive binary search, and can never disagree with what regression_coefs.py
actually fit. In epv mode, this sweeps multiple epv values for one tissue
(comparing epv choices is this script's purpose there, unlike looping over
tissues or models, which Snakemake fans out over) and produces a
multi-row comparison table. In binsearch mode there is no free parameter to
sweep - exactly one feature set exists per tissue - so it evaluates that
one set and produces a single-row table in the same schema.

Inputs:
  - results/regression_coefs/<shap_model>/<selection_tag>/<tissue>_summary.csv
    (one per epv value in epv mode, or the single binsearch one)
  - results/training_data/unfiltered/hrs<window>/motif_enrichment_hrs<window>.csv
  - results/training_data/unfiltered/hrs<window>/y_<tissue>.csv

Outputs:
  - results/logit_regression_cv_aucroc/<shap_model>/<tissue>_logit_cv_results_epv_sweep.csv (epv mode)
  - results/logit_regression_cv_aucroc/<shap_model>/<tissue>_logit_cv_results_binsearch.csv (binsearch mode)
"""

import argparse
import logging
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from core.constants import MODELS, FeatureMode, FeatureSelectionMode
from core.features import stack_windows
from core.log import configure_logging
from core.logit_analysis import logit_cross_validate
from first_logit_model.features import feature_selection_tag
from first_logit_model.io import extract_used_features

logger = logging.getLogger(__name__)


def evaluate_selection(
    tissue: str, shap_model: str, selection_tag: str, selection_value, n_splits: int, p_value_threshold: float
) -> dict:
    """
    Cross-validate regression_coefs.py's already-chosen feature set for one
    (tissue, shap_model, selection_tag).

    selection_tag drives the summary-file lookup (e.g. "epv_10"); selection_value
    is what's reported in the output table's "selection" column -- the plain
    epv number in epv mode, so the column stays numeric, or "binsearch" in
    binsearch mode (no numeric equivalent there).

    Returns a dict with the CV AUC and the BH-significant feature count
    from regression_coefs.py's own full-data fit for the same combination.
    """
    features = extract_used_features(tissue, shap_model, selection_tag)
    num_features = len(features)
    logger.info(f"[{tissue}] shap_model={shap_model}, selection={selection_tag}, num_features={num_features}")

    X, y, composite = stack_windows(tissue, feature_mode=FeatureMode.CURRENT)
    X = X[features]

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    mean_auc, std_auc = logit_cross_validate(X, y, splitter, stratify_target=composite)

    summary_path = f"results/regression_coefs/{shap_model}/{selection_tag}/{tissue}_summary.csv"
    regression_summary = pd.read_csv(summary_path)

    num_significant_bh = (regression_summary["p_adjusted_bh"] < p_value_threshold).sum()

    logger.info(
        f"[{tissue}] AUC={mean_auc:.4f} ± {std_auc:.4f}, "
        f"significant_features={num_significant_bh} (Benjamini-Hochberg)"
    )

    return {
        "tissue": tissue,
        "selection": selection_value,
        "num_features": num_features,
        "num_features_significant_bh": num_significant_bh,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-validate the first logit model's chosen feature set(s) for one tissue"
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
        "--epv_values", type=int, nargs="+", default=None,
        help="Required when --feature_selection_mode is epv; ignored otherwise",
    )
    parser.add_argument("--n_splits", type=int, required=True)
    parser.add_argument("--p_value_threshold", type=float, required=True)
    parser.add_argument("--log_path", type=Path, required=True)
    args = parser.parse_args()

    if args.feature_selection_mode == FeatureSelectionMode.EPV and not args.epv_values:
        parser.error("--epv_values is required when --feature_selection_mode is epv")

    configure_logging(args.log_path)

    if args.feature_selection_mode == FeatureSelectionMode.EPV:
        selections = [(feature_selection_tag(FeatureSelectionMode.EPV, epv), epv) for epv in args.epv_values]
    else:
        selections = [(feature_selection_tag(FeatureSelectionMode.BINSEARCH, None), "binsearch")]

    all_results = []
    for selection_tag, selection_value in selections:
        try:
            all_results.append(
                evaluate_selection(
                    args.tissue, args.shap_model, selection_tag, selection_value,
                    args.n_splits, args.p_value_threshold,
                )
            )
        except Exception as e:
            logger.info(f"[{args.tissue}] {selection_tag} FAILED: {e}")
            continue

    mode_tag = "epv_sweep" if args.feature_selection_mode == FeatureSelectionMode.EPV else "binsearch"

    output_dir = f"results/logit_regression_cv_aucroc/{args.shap_model}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{args.tissue}_logit_cv_results_{mode_tag}.csv"
    columns = ["selection", "num_features", "num_features_significant_bh", "mean_auc", "std_auc"]

    if not all_results:
        # every selection_tag's CV failed (e.g. a singular matrix in some
        # fold) - still write the declared output (empty but headed) so a
        # legitimate "nothing evaluated cleanly" result is distinguishable
        # from the file simply never having been produced.
        logger.info(f"No successful models for tissue: {args.tissue}")
        pd.DataFrame(columns=columns).to_csv(output_path, index=False)
        return

    results_df = pd.DataFrame(all_results)
    results_df["mean_auc"] = results_df["mean_auc"].round(6)
    results_df["std_auc"] = results_df["std_auc"].round(6)
    results_df = results_df.sort_values("mean_auc", ascending=False)
    results_df = results_df[columns]
    results_df.to_csv(output_path, index=False)
    logger.info(f"Table for tissue {args.tissue} saved at {output_path}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
