# logit_analysis.py

"""
Shared statsmodels Logit fitting, cross-validation, and feature-count search
for the logit-model stages (first_logit_model, second_logit_model).

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
from statsmodels.tools.sm_exceptions import PerfectSeparationError


def fit_logit_summary(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Fit a statsmodels Logit on the full data and summarize its coefficients.

    Returns a DataFrame indexed by feature name (plus "const" for the
    intercept) with coef, std_err, z, unadjusted/BH-adjusted/TSBH-adjusted
    p-values, and 95% CI bounds, sorted by absolute coefficient size.
    """
    res = sm.Logit(y, sm.add_constant(X)).fit(disp=False)

    p_unadjusted = res.pvalues

    # a handful of coefficients routinely come out with a NaN Wald p-value
    # (near-singular variance estimate from collinear motif features) -
    # multipletests() propagates a single NaN to its *entire* output array,
    # so it has to run on the finite subset only. NaN stays NaN (correctly
    # non-significant) for those features.
    finite = p_unadjusted.notna()
    p_bh = pd.Series(np.nan, index=p_unadjusted.index)
    p_tsbh = pd.Series(np.nan, index=p_unadjusted.index)
    if finite.any():
        _, p_bh[finite], _, _ = multipletests(p_unadjusted[finite], method="fdr_bh")
        _, p_tsbh[finite], _, _ = multipletests(p_unadjusted[finite], method="fdr_tsbh")

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


def find_max_features_binary_search(X_ranked: pd.DataFrame, y: pd.Series) -> int:
    """
    Binary search the largest prefix of X_ranked's columns that fits a
    statsmodels Logit without a singular-matrix / perfect-separation error.

    X_ranked's columns must already be sorted by descending importance -
    this searches prefixes, not arbitrary subsets. Assumes monotonicity:
    if N features fit, every N' < N also fits.
    """
    lo, hi, best = 1, X_ranked.shape[1], 1

    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            sm.Logit(y, sm.add_constant(X_ranked.iloc[:, :mid])).fit(disp=False)
        except (np.linalg.LinAlgError, PerfectSeparationError):
            hi = mid - 1
            continue
        best, lo = mid, mid + 1

    return best


def logit_cross_validate(X: pd.DataFrame, y: pd.Series, splitter, stratify_target) -> tuple[float, float]:
    """
    Cross-validate a statsmodels Logit and return (mean_auc, std_auc).

    Args:
        X, y: feature matrix and binary target.
        splitter: an already-configured StratifiedKFold instance.
        stratify_target: passed to splitter.split(X, stratify_target) -
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
