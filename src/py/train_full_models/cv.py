# cv.py

"""
Shared cross-validation harness for all training modes in this stage
(time-specific, time-agnostic, expanded time-agnostic).

Pipeline context: used by train_time_specific.py and train_time_agnostic.py.
Both scripts build their own (classifier, X, y, splitter) and hand them to
cross_validate(), so the CV loop, ROC-curve aggregation, and full-data model
fit are only implemented once. This module has no CLI entry point.

Inputs: none (operates on in-memory arrays passed by the caller).
Outputs: none (returns a CVResult / fitted model; callers persist results).
"""

from dataclasses import dataclass

import numpy as np
from sklearn.base import clone
from sklearn.metrics import roc_curve, auc, accuracy_score


@dataclass
class CVResult:
    """Aggregated cross-validation metrics and mean ROC curve."""

    mean_auc: float
    std_auc: float
    mean_acc: float
    std_acc: float
    mean_fpr: np.ndarray
    mean_tpr: np.ndarray
    tprs_upper: np.ndarray
    tprs_lower: np.ndarray
    # per-fold AUCs and (train_idx, test_idx) pairs, needed by
    # train_time_specific.py to pick and retrain the single best fold
    roc_aucs: list
    fold_indices: list


def cross_validate(classifier, X, y, splitter, stratify_target) -> CVResult:
    """
    Run cross-validation with the given splitter and aggregate per-fold ROC
    curves into one mean ROC curve with a +/- 1 std. dev. band.

    Args:
        classifier: unfitted sklearn-style estimator, cloned per fold.
        X, y: feature matrix and binary target.
        splitter: an already-configured KFold or StratifiedKFold instance.
        stratify_target: passed to splitter.split(X, stratify_target) -
            the composite stratification array for StratifiedKFold, or None
            for plain KFold (which ignores it).

    Returns:
        CVResult with aggregated metrics and the mean ROC curve.
    """
    probs, trues, accs, fold_indices = [], [], [], []

    for train_idx, test_idx in splitter.split(X, stratify_target):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        fold_indices.append((train_idx, test_idx))

        clf = clone(classifier)
        clf.fit(X_train, y_train)

        p = clf.predict_proba(X_test)[:, 1]
        probs.append(p)
        trues.append(y_test.values)
        accs.append(accuracy_score(y_test, (p > 0.5).astype(int)))

    mean_acc = np.mean(accs)
    std_acc = np.std(accs)

    fprs, tprs, roc_aucs = [], [], []
    for true, prob in zip(trues, probs):
        fpr, tpr, _ = roc_curve(true, prob)
        fprs.append(fpr)
        tprs.append(tpr)
        roc_aucs.append(auc(fpr, tpr))

    n_splits = splitter.get_n_splits()
    mean_fpr = np.linspace(0, 1, 100)
    interp_tprs = []
    for i in range(n_splits):
        interp_tpr = np.interp(mean_fpr, fprs[i], tprs[i])
        interp_tpr[0] = 0.0
        interp_tprs.append(interp_tpr)

    mean_tpr = np.mean(interp_tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(roc_aucs)
    tprs_upper = np.minimum(mean_tpr + np.std(interp_tprs, axis=0), 1)
    tprs_lower = np.maximum(mean_tpr - np.std(interp_tprs, axis=0), 0)

    return CVResult(
        mean_auc=float(mean_auc),
        std_auc=float(std_auc),
        mean_acc=float(mean_acc),
        std_acc=float(std_acc),
        mean_fpr=mean_fpr,
        mean_tpr=mean_tpr,
        tprs_upper=tprs_upper,
        tprs_lower=tprs_lower,
        roc_aucs=roc_aucs,
        fold_indices=fold_indices,
    )


def fit_full_data_model(classifier, X, y):
    """Fit a fresh clone of classifier on the full dataset (no held-out fold)."""
    clf = clone(classifier)
    clf.fit(X, y)
    return clf
