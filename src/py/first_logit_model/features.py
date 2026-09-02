# features.py

"""
Feature-count selection (EPV or binary search) and SHAP-ranked feature
downsampling for the first logit model stage.

Pipeline context: used by regression_coefs.py and logit_regression_aucroc.py
to decide how many motifs to preselect, and which ones, before fitting a
statsmodels Logit on core.features.stack_windows' full motif set.
feature_selection_tag() builds the <selection_tag> path segment that both
scripts, plus second_logit_model, use to name their outputs.

Inputs:
  - results/prepare_data/unfiltered/hrs<window>/y_<tissue>.csv
  - results/shap_importance/shap_analysis/tables/<shap_model>/<tissue>.csv

Outputs: none (returns computed values in memory).
"""

import os

import numpy as np
import pandas as pd

from core.constants import WINDOWS, FeatureSelectionMode
from core.paths import prepare_data_dir, shap_table_path


def num_features_from_epv(tissue: str, epv: int) -> int:
    """
    Compute the EPV-based feature budget for one tissue: the number of
    positive-class loops (events) across all three windows, divided by epv.

    events per variable: EPV = number_of_events / number_of_features
    """
    training_data_dir = prepare_data_dir("unfiltered")
    num_events = 0

    for w in WINDOWS:
        y = pd.read_csv(
            os.path.join(training_data_dir, f"hrs{w}/y_{tissue}.csv"), index_col=0
        ).iloc[:, 0]
        num_events += y.sum()

    return int(np.ceil(num_events / epv))


def downsample_features_shap(tissue: str, shap_model: str, num_features: int | None = None) -> pd.Series:
    """
    Return motif IDs ranked by mean absolute SHAP importance, descending.

    With num_features given, returns only the top num_features. With
    num_features=None, returns the full ranking (e.g. for binary search,
    which needs the whole ordering to take prefixes of).
    """
    shap_table = pd.read_csv(shap_table_path(shap_model, tissue))
    sorted_features = shap_table.sort_values("abs_mean_importance", ascending=False)["motif_id"]
    return sorted_features[:num_features] if num_features is not None else sorted_features


def feature_selection_tag(mode: FeatureSelectionMode, epv: int | None) -> str:
    """
    Build the <selection_tag> path segment for the first/second logit model
    stages' output paths: "epv_<epv>" for EPV mode, "binsearch" otherwise.
    """
    if mode == FeatureSelectionMode.EPV:
        if epv is None:
            raise ValueError("epv is required when feature_selection_mode is EPV")
        return f"epv_{epv}"
    return "binsearch"
