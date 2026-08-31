# features.py

"""
EPV-based feature budgeting and SHAP-ranked feature downsampling for the
first logit model stage.

Pipeline context: used by regression_coefs.py and logit_regression_aucroc.py
to decide how many motifs to preselect, and which ones, before fitting a
statsmodels Logit on core.features.stack_windows' full motif set.

Inputs:
  - results/training_data/unfiltered/hrs<window>/y_<tissue>.csv
  - results/shap/<shap_model>/<tissue>_shap_table_<shap_model>.csv

Outputs: none (returns computed values in memory).
"""

import os

import numpy as np
import pandas as pd

from core.constants import WINDOWS


def num_features_from_epv(tissue: str, epv: int) -> int:
    """
    Compute the EPV-based feature budget for one tissue: the number of
    positive-class loops (events) across all three windows, divided by epv.

    events per variable: EPV = number_of_events / number_of_features
    """
    training_data_dir = "results/training_data/unfiltered"
    num_events = 0

    for w in WINDOWS:
        y = pd.read_csv(
            os.path.join(training_data_dir, f"hrs{w}/y_{tissue}.csv"), index_col=0
        ).iloc[:, 0]
        num_events += y.sum()

    return int(np.ceil(num_events / epv))


def downsample_features_shap(tissue: str, shap_model: str, num_features: int) -> pd.Series:
    """Return the top num_features motif IDs ranked by mean absolute SHAP importance."""
    shap_table = pd.read_csv(f"results/shap/{shap_model}/{tissue}_shap_table_{shap_model}.csv")
    sorted_features = shap_table.sort_values("abs_mean_importance", ascending=False)["motif_id"]
    return sorted_features[:num_features]
