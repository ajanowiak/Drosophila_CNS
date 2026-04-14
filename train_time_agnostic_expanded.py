import os
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, accuracy_score
from xgboost import XGBClassifier

from utils import print_timestamp


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WINDOWS = ["06-08", "10-12", "14-16"]
TISSUES = ["Neuroblasts", "Neurons", "Glia"]
PREV_WINDOW = {
    "06-08": "04-06",
    "10-12": "08-10",
    "14-16": "12-14",
}

N_SPLITS = 10

MODEL_PARAMS = {
    "RandomForestClassifier": dict(n_estimators=500, random_state=0, n_jobs=4),
    "SVC":                    dict(probability=True, random_state=0),
    "LogisticRegression":     dict(max_iter=1000, random_state=0, n_jobs=4, C=np.inf), #UNREGULARIZED
    "XGBClassifier":          dict(
        n_estimators=500,
        random_state=0,
        n_jobs=4,
    ),
}

MODEL_CLASSES = {
    "RandomForestClassifier": RandomForestClassifier,
    "SVC":                    SVC,
    "LogisticRegression":     LogisticRegression,
    "XGBClassifier":          XGBClassifier,
}


def make_names_dict():
    """
    Dictionary for f-string construction in model paths and in text on plots.
    """
    model_names = [
        "RandomForestClassifier",
        "SVC",
        "LogisticRegression",
        "XGBClassifier",
    ]
    model_names_dict = {name: {"full": "", "short": ""} for name in model_names}
    model_names_dict["RandomForestClassifier"]["full"]  = "Random Forest"
    model_names_dict["RandomForestClassifier"]["short"] = "RF"
    model_names_dict["SVC"]["full"]                     = "Support Vector Machine"
    model_names_dict["SVC"]["short"]                    = "SVM"
    model_names_dict["LogisticRegression"]["full"]      = "Logistic Regression"
    model_names_dict["LogisticRegression"]["short"]     = "LR"
    model_names_dict["XGBClassifier"]["full"]           = "XGBoost"
    model_names_dict["XGBClassifier"]["short"]          = "XGB"
    return model_names_dict


NAMES = make_names_dict()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_enrichment(feature_dir_template: str, window: str) -> pd.DataFrame:
    """
    Load the motif enrichment matrix for a single window.

    Args:
        feature_dir_template: f-string template with {window} placeholder.
        window: window tag, e.g. "06-08" or "04-06".

    Returns:
        DataFrame with loops as rows and TF motifs as columns.
    """
    feature_dir = feature_dir_template.format(window=window)
    path = os.path.join(feature_dir, f"motif_enrichment_hrs{window}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Enrichment file not found: {path}")
    return pd.read_csv(path, index_col=0)


def _load_labels(label_tag: str, window: str, tissue: str) -> pd.Series:
    """
    Load the binary target vector for a single tissue / window pair.

    Args:
        label_tag: "neural_labels" or "unfiltered".
        window:    window tag, e.g. "06-08".
        tissue:    tissue name, e.g. "Neuroblasts".

    Returns:
        Series indexed by loop ID.
    """
    path = f"results/training_data/{label_tag}/hrs{window}/y_{tissue}.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Label file not found: {path}")
    return pd.read_csv(path, index_col=0).iloc[:, 0]


def compose_windows_expanded(
    tissue: str,
    feature_dir_template: str,
    label_tag: str,
) -> tuple:
    """
    Build the expanded feature matrix by horizontally concatenating current-
    and previous-window enrichment scores for each developmental window.

    Feature naming convention
    -------------------------
    All columns from the current window are suffixed ``_curr``;
    all columns from the predecessor window are suffixed ``_prev``.

    Alignment strategy
    ------------------
    For each main window w:
      1. Load X_curr (current enrichment) and y (labels) — align on their
         shared loop index, then drop NaN rows from X_curr.
      2. Load X_prev (predecessor enrichment) independently.
      3. Take the three-way intersection of loop indices: X_curr ∩ y ∩ X_prev.
         Loops missing from *any* of the three are silently dropped.
      4. Drop rows that still contain NaN in either enrichment matrix.
      5. Rename columns and concatenate horizontally.

    Args:
        tissue:                tissue label, e.g. "Neuroblasts".
        feature_dir_template:  f-string template with {window} placeholder.
        label_tag:             "neural_labels" or "unfiltered".

    Returns:
        X          (pd.DataFrame): shape (n_samples, 722), columns suffixed
                                   _curr / _prev.
        y          (pd.Series):    binary target, same index as X.
        composite  (np.ndarray):   integer codes for composite stratification
                                   (window_index × label), used by
                                   StratifiedKFold to preserve both the
                                   window origin and class balance in every
                                   fold.
    """
    Xs, ys = [], []

    for idx, w in enumerate(WINDOWS):
        prev_w = PREV_WINDOW[w]

        X_curr = _load_enrichment(feature_dir_template, w)
        X_prev = _load_enrichment(feature_dir_template, prev_w)
        y_w    = _load_labels(label_tag, w, tissue)

        shared = X_curr.index.intersection(y_w.index).intersection(X_prev.index)

        n_curr_only = len(X_curr.index.intersection(y_w.index)) - len(shared)
        if n_curr_only > 0:
            print_timestamp(
                f"  [{tissue}] hrs{w}: {n_curr_only} loops absent from "
                f"prev window ({prev_w}) — excluded from expanded dataset"
            )

        X_curr = X_curr.loc[shared]
        X_prev = X_prev.loc[shared]
        y_w    = y_w.loc[shared]

        nan_mask = X_curr.isna().any(axis=1) | X_prev.isna().any(axis=1)
        n_dropped = nan_mask.sum()
        if n_dropped > 0:
            print_timestamp(
                f"  [{tissue}] hrs{w}: dropped {n_dropped} loops with NaN "
                f"in current or previous enrichment "
                f"({(~nan_mask).sum()} remaining)"
            )
        X_curr = X_curr.loc[~nan_mask]
        X_prev = X_prev.loc[~nan_mask]
        y_w    = y_w.loc[~nan_mask]

        X_curr = X_curr.add_suffix("_curr")
        X_prev = X_prev.add_suffix("_prev")

        X_w = pd.concat([X_curr, X_prev], axis=1)

        assert X_w.shape[1] == X_curr.shape[1] + X_prev.shape[1], (
            f"Column count mismatch after concat for {tissue} / hrs{w}"
        )

        X_w["_window"] = idx
        Xs.append(X_w)
        ys.append(y_w)

    X = pd.concat(Xs, axis=0)
    y = pd.concat(ys, axis=0)

    composite = pd.Categorical(list(zip(X["_window"], y))).codes
    X = X.drop(columns=["_window"])

    print_timestamp(
        f"[{tissue}] Expanded feature matrix: {X.shape} | "
        f"positives: {y.sum()} / {len(y)}"
    )

    return X, y, composite


# ---------------------------------------------------------------------------
# Training for a single tissue
# ---------------------------------------------------------------------------

def train_tissue_expanded(
    tissue: str,
    feature_dir_template: str,
    label_tag: str,
    model_name: str,
    n_splits: int = N_SPLITS,
) -> dict:
    """
    Cross-validated time-agnostic training with expanded (curr + prev)
    features for one tissue, supporting any of the four model architectures.

    Args:
        tissue:                tissue label.
        feature_dir_template:  f-string template with {window} placeholder.
        label_tag:             "neural_labels" or "unfiltered".
        model_name:            one of the keys in MODEL_CLASSES.
        n_splits:              number of CV folds.

    Returns:
        dict with tissue name and CV metrics (AUC, accuracy, ROC curve data).
    """
    short = NAMES[model_name]["short"]
    full  = NAMES[model_name]["full"]

    print_timestamp(
        f"[{tissue}] Starting EXPANDED training — {full} (label_tag={label_tag})..."
    )

    X, y, composite = compose_windows_expanded(
        tissue, feature_dir_template, label_tag
    )

    classifier = MODEL_CLASSES[model_name](**MODEL_PARAMS[model_name])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

    probs, trues, accs = [], [], []

    for train_idx, test_idx in skf.split(X, composite):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        clf = clone(classifier)
        clf.fit(X_train, y_train)

        p = clf.predict_proba(X_test)[:, 1]
        probs.append(p)
        trues.append(y_test.values)
        accs.append(accuracy_score(y_test, (p > 0.5).astype(int)))

    # --- ROC metrics ---
    mean_acc = np.mean(accs)
    std_acc  = np.std(accs)

    fprs, tprs, roc_aucs = [], [], []
    for true, prob in zip(trues, probs):
        fpr, tpr, _ = roc_curve(true, prob)
        fprs.append(fpr)
        tprs.append(tpr)
        roc_aucs.append(auc(fpr, tpr))

    mean_fpr = np.linspace(0, 1, 100)
    interp_tprs = []
    for i in range(n_splits):
        interp_tpr = np.interp(mean_fpr, fprs[i], tprs[i])
        interp_tpr[0] = 0.0
        interp_tprs.append(interp_tpr)

    mean_tpr     = np.mean(interp_tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc     = auc(mean_fpr, mean_tpr)
    std_auc      = np.std(roc_aucs)
    tprs_upper   = np.minimum(mean_tpr + np.std(interp_tprs, axis=0), 1)
    tprs_lower   = np.maximum(mean_tpr - np.std(interp_tprs, axis=0), 0)

    # --- ROC figure ---
    _, ax = plt.subplots(figsize=(6, 6))
    ax.plot(
        mean_fpr, mean_tpr,
        label=f"Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})",
        lw=1, alpha=0.8, color="firebrick",
    )
    ax.fill_between(
        mean_fpr, tprs_lower, tprs_upper, alpha=0.2, label="± 1 std. dev.", color="firebrick",
    )
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.grid(axis="both")
    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=(
            f"Time-agnostic {full} ROC — expanded features ({tissue})\n"
            f"AUC = {mean_auc:.3f} ± {std_auc:.3f} \n"
            f"Acc = {mean_acc:.3f} ± {std_acc:.3f}"
        ),
    )
    ax.legend(loc="lower right")

    fig_dir = (
        f"results/figures/time_agnostic_EXPANDED"
        f"{short}_expanded"
    )
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, f"roc_{short}_expanded_{tissue}.pdf")
    plt.savefig(fig_path, dpi=300, format="pdf", bbox_inches="tight")
    plt.close()
    print_timestamp(f"[{tissue}] ROC figure saved to {fig_path}")

    # --- Save full-data model (trained on all samples, used for scoring) ---
    all_clf = clone(classifier)
    all_clf.fit(X, y)

    all_dir  = f"results/models/time_agnostic_expanded/{short}"
    os.makedirs(all_dir, exist_ok=True)
    all_path = os.path.join(all_dir, f"{short}_{tissue}_expanded.pkl")
    pickle.dump(all_clf, open(all_path, "wb"))
    print_timestamp(f"[{tissue}] Full-data model saved to {all_path}")

    print_timestamp(
        f"[{tissue}] Done — mean AUC={mean_auc:.4f} ± {std_auc:.4f}, "
        f"mean Acc={mean_acc:.4f} ± {std_acc:.4f}"
    )

    return {
        "tissue":     tissue,
        "mean_auc":   mean_auc,
        "std_auc":    std_auc,
        "mean_acc":   mean_acc,
        "std_acc":    std_acc,
        "mean_fpr":   mean_fpr,
        "mean_tpr":   mean_tpr,
        "tprs_upper": tprs_upper,
        "tprs_lower": tprs_lower,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train time-agnostic classifiers with expanded features "
            "(current + previous window motif enrichment)."
        )
    )
    parser.add_argument(
        "--filter_labels",
        type=lambda x: x.lower() == "true",
        default=False,
        help=(
            "Use neural-label-filtered enrichment (True) or "
            "unfiltered (False)"
            "(default: False)"
        ),
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_CLASSES.keys()),
        default="RandomForestClassifier",
        help=f"Model architecture to train (default: RandomForestClassifier). \n Choices: {list(MODEL_CLASSES.keys())}",
    )
    args = parser.parse_args()

    label_tag  = "neural_labels" if args.filter_labels else "unfiltered"
    model_name = args.model
    short      = NAMES[model_name]["short"]
    full       = NAMES[model_name]["full"]

    feature_dir_template = f"results/training_data/{label_tag}/hrs{{window}}"

    print_timestamp(
        f"=== Training time-agnostic {full} (EXPANDED) | features: {label_tag} ==="
    )
    print_timestamp(f"Feature directory template: {feature_dir_template}")
    print_timestamp(
        "Window → predecessor map: "
        + ", ".join(f"{k}→{v}" for k, v in PREV_WINDOW.items())
    )

    # Process tissues in parallel (one worker per tissue)
    results = {}
    with ProcessPoolExecutor(max_workers=len(TISSUES)) as executor:
        futures = {
            executor.submit(
                train_tissue_expanded,
                t, feature_dir_template, label_tag, model_name, N_SPLITS,
            ): t
            for t in TISSUES
        }
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                results[t] = fut.result()
            except Exception as e:
                print_timestamp(f"[{t}] FAILED: {e}")
                raise

    # --- Summary table ---
    rows = []
    for t in TISSUES:
        if t not in results:
            continue
        r = results[t]
        rows.append({
            "tissue":    t,
            "model":     model_name,
            "label_tag": label_tag,
            "features":  "curr+prev",
            "mean_auc":  round(r["mean_auc"], 6),
            "std_auc":   round(r["std_auc"],  6),
            "mean_acc":  round(r["mean_acc"], 6),
            "std_acc":   round(r["std_acc"],  6),
        })

    summary_df   = pd.DataFrame(rows)
    summary_dir  = f"results/time_agnostic_expanded/{label_tag}/{short}"
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(
        summary_dir, f"cv_aucroc_summary_{short}_expanded.csv"
    )
    summary_df.to_csv(summary_path, index=False)
    print_timestamp(f"Summary table saved to {summary_path}")

    print("\n" + summary_df.to_string(index=False))
    print_timestamp("=== All done ===")


if __name__ == "__main__":
    main()