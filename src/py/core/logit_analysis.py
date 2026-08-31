# logit_analysis.py

"""
Shared statsmodels Logit fitting and cross-validation for the logit-model
stages (first_logit_model, second_logit_model).

Pipeline context: both stages fit a statsmodels Logit on a downsampled
feature set and estimate CV AUC the same way; this module implements that
once. Not reusable with train_full_models.cv (sklearn-based, expects a
classifier with .fit()/.predict_proba()) since statsmodels' Logit has a
different fitting API.

Inputs: none (operates on in-memory arrays passed by the caller).
Outputs: none (returns a summary DataFrame / AUC stats; callers persist results).
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_curve, auc
from statsmodels.stats.multitest import multipletests


def fit_logit_summary(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Fit a statsmodels Logit on the full data and summarize its coefficients.

    Returns a DataFrame indexed by feature name (plus "const" for the
    intercept) with coef, std_err, z, unadjusted/BH-adjusted/TSBH-adjusted
    p-values, and 95% CI bounds, sorted by absolute coefficient size.
    """
    res = sm.Logit(y, sm.add_constant(X)).fit(disp=False)

    p_unadjusted = res.pvalues
    _, p_bh, _, _ = multipletests(p_unadjusted, method="fdr_bh")
    _, p_tsbh, _, _ = multipletests(p_unadjusted, method="fdr_tsbh")

    summary_df = pd.DataFrame({
        "coef": res.params,
        "std_err": res.bse,
        "z": res.tvalues,
        "p_unadjusted": p_unadjusted,
        "p_adjusted_bh": p_bh,
        "p_adjusted_tsbh": p_tsbh,
    })

    ci = res.conf_int()
    ci.columns = ["ci_lower", "ci_upper"]
    summary_df = pd.concat([summary_df, ci], axis=1)

    return summary_df.sort_values("coef", key=abs, ascending=False)


def logit_cross_validate(X: pd.DataFrame, y: pd.Series, splitter, stratify_target) -> tuple[float, float]:
    """
    Cross-validate a statsmodels Logit and return (mean_auc, std_auc).

    Args:
        X, y: feature matrix and binary target.
        splitter: an already-configured StratifiedKFold instance.
        stratify_target: passed to splitter.split(X, stratify_target) --
            the composite stratification array.
    """
    roc_aucs = []

    for train_idx, test_idx in splitter.split(X, stratify_target):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        res_train = sm.Logit(y_train, sm.add_constant(X_train)).fit(disp=False)
        probs = res_train.predict(sm.add_constant(X_test))

        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_aucs.append(auc(fpr, tpr))

    return float(np.mean(roc_aucs)), float(np.std(roc_aucs))
