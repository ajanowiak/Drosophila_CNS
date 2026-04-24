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

from utils import print_timestamp, make_names_dict


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
    "LogisticRegression":     dict(max_iter=1000, random_state=0, n_jobs=4),
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

# Color assigned to each feature-window mode in ROC curve plots
FEATURE_MODE_COLORS = {
    "curr":      "steelblue",
    "prev":      "grey",
    "curr+prev": "firebrick",
}

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


def compose_windows(
    tissue: str,
    feature_dir_template: str,
    label_tag: str,
    feature_mode: str,
) -> tuple:
    """
    Build the feature matrix for the requested feature_mode across all
    developmental windows, stacking them vertically for time-agnostic training.

    Feature modes
    -------------
    "curr"     : only current-window enrichment scores (columns suffixed _curr).
    "prev"     : only previous-window enrichment scores (columns suffixed _prev).
    "curr+prev": horizontal concatenation of both (columns suffixed accordingly).

    Alignment strategy
    ------------------
    For each main window w:
      1. Load X_curr and y — align on shared loop index, drop NaN rows.
      2. If prev features are needed, load X_prev and take the three-way
         intersection of loop indices (X_curr ∩ y ∩ X_prev). Loops missing
         from any source are silently dropped.
      3. Drop rows that still contain NaN in any loaded enrichment matrix.
      4. Rename columns and concatenate horizontally (if both modes).

    Args:
        tissue:                tissue label, e.g. "Neuroblasts".
        feature_dir_template:  f-string template with {window} placeholder.
        label_tag:             "neural_labels" or "unfiltered".
        feature_mode:          one of "curr", "prev", or "curr+prev".

    Returns:
        X          (pd.DataFrame): feature matrix.
        y          (pd.Series):    binary target, same index as X.
        composite  (np.ndarray):   integer codes for composite stratification
                                   (window_index × label), used by
                                   StratifiedKFold to preserve both the
                                   window origin and class balance in every fold.
    """
    use_curr = feature_mode in ("curr", "curr+prev")
    use_prev = feature_mode in ("prev", "curr+prev")

    Xs, ys = [], []

    for idx, w in enumerate(WINDOWS):
        prev_w = PREV_WINDOW[w]

        X_curr = _load_enrichment(feature_dir_template, w)
        y_w    = _load_labels(label_tag, w, tissue)

        # Base shared index: curr ∩ labels
        shared = X_curr.index.intersection(y_w.index)

        # Extend intersection to prev if needed
        if use_prev:
            X_prev = _load_enrichment(feature_dir_template, prev_w)
            shared = shared.intersection(X_prev.index)

            n_curr_only = len(X_curr.index.intersection(y_w.index)) - len(shared)
            if n_curr_only > 0:
                print_timestamp(
                    f"  [{tissue}] hrs{w}: {n_curr_only} loops absent from "
                    f"prev window ({prev_w}) — excluded from dataset"
                )

        X_curr = X_curr.loc[shared]
        y_w    = y_w.loc[shared]

        # Build NaN mask across whichever matrices are loaded
        nan_mask = X_curr.isna().any(axis=1)
        if use_prev:
            X_prev = X_prev.loc[shared]
            nan_mask = nan_mask | X_prev.isna().any(axis=1)

        n_dropped = nan_mask.sum()
        if n_dropped > 0:
            print_timestamp(
                f"  [{tissue}] hrs{w}: dropped {n_dropped} loops with NaN "
                f"({(~nan_mask).sum()} remaining)"
            )

        X_curr = X_curr.loc[~nan_mask]
        y_w    = y_w.loc[~nan_mask]

        # Assemble feature block for this window
        parts = []
        if use_curr:
            parts.append(X_curr.add_suffix("_curr"))
        if use_prev:
            parts.append(X_prev.loc[~nan_mask].add_suffix("_prev"))

        X_w = pd.concat(parts, axis=1)
        X_w["_window"] = idx
        Xs.append(X_w)
        ys.append(y_w)

    X = pd.concat(Xs, axis=0)
    y = pd.concat(ys, axis=0)

    composite = pd.Categorical(list(zip(X["_window"], y))).codes
    X = X.drop(columns=["_window"])

    print_timestamp(
        f"[{tissue}] Feature matrix ({feature_mode}): {X.shape} | "
        f"positives: {y.sum()} / {len(y)}"
    )

    return X, y, composite


# ---------------------------------------------------------------------------
# Training for a single tissue
# ---------------------------------------------------------------------------

def train_tissue(
    tissue: str,
    feature_dir_template: str,
    label_tag: str,
    model_name: str,
    feature_mode: str,
    n_splits: int = N_SPLITS,
) -> dict:
    """
    Cross-validated time-agnostic training for one tissue.

    Args:
        tissue:                tissue label.
        feature_dir_template:  f-string template with {window} placeholder.
        label_tag:             "neural_labels" or "unfiltered".
        model_name:            one of the keys in MODEL_CLASSES.
        feature_mode:          one of "curr", "prev", or "curr+prev".
        n_splits:              number of CV folds.

    Returns:
        dict with tissue name and CV metrics (AUC, accuracy, ROC curve data).
    """
    short  = NAMES[model_name]["short"]
    full   = NAMES[model_name]["full"]
    color  = FEATURE_MODE_COLORS[feature_mode]

    print_timestamp(
        f"[{tissue}] Starting training — {full} | features={feature_mode} "
        f"(label_tag={label_tag})..."
    )

    X, y, composite = compose_windows(
        tissue, feature_dir_template, label_tag, feature_mode
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

    # --- ROC figure (saved as both PDF and PNG) ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(
        mean_fpr, mean_tpr,
        label=f"Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})",
        lw=1, alpha=0.8, color=color,
    )
    ax.fill_between(
        mean_fpr, tprs_lower, tprs_upper,
        alpha=0.2, label="± 1 std. dev.", color=color,
    )
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.grid(axis="both")
    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=(
            f"Time-agnostic {full} ROC: {feature_mode} features ({tissue})\n"
            f"AUC = {mean_auc:.3f} ± {std_auc:.3f} \n"
            f"Acc = {mean_acc:.3f} ± {std_acc:.3f}"
        ),
    )
    ax.legend(loc="lower right")

    # Use a filesystem-safe mode tag (replace "+" with "plus")
    mode_tag = feature_mode.replace("+", "plus")

    fig_dir = f"results/figures/time_agnostic/{short}_{mode_tag}"
    os.makedirs(fig_dir, exist_ok=True)

    for fmt in ("pdf", "png"):
        fig_path = os.path.join(
            fig_dir, f"roc_{short}_{mode_tag}_{tissue}.{fmt}"
        )
        fig.savefig(fig_path, dpi=300, format=fmt, bbox_inches="tight")
        print_timestamp(f"[{tissue}] ROC figure ({fmt}) saved to {fig_path}")

    plt.close(fig)

    # --- Save full-data model (trained on all samples, used for scoring) ---
    all_clf = clone(classifier)
    all_clf.fit(X, y)

    model_dir = f"results/models/time_agnostic/{short}/{mode_tag}"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{short}_{tissue}_{mode_tag}.pkl")
    pickle.dump(all_clf, open(model_path, "wb"))
    print_timestamp(f"[{tissue}] Full-data model saved to {model_path}")

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
            "Train time-agnostic classifiers with selectable feature windows "
            "(current, previous, or both) and model architecture."
        )
    )
    parser.add_argument(
        "--filter_labels",
        type=lambda x: x.lower() == "true",
        default=False,
        help=(
            "Use neural-label-filtered enrichment (True) or unfiltered (False) "
            "(default: False)"
        ),
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_CLASSES.keys()),
        default="RandomForestClassifier",
        help=(
            "Model architecture to train (default: RandomForestClassifier). "
            f"Choices: {list(MODEL_CLASSES.keys())}"
        ),
    )
    parser.add_argument(
        "--feature_mode",
        choices=["curr", "prev", "curr+prev"],
        default="curr+prev",
        help=(
            "Which enrichment window(s) to use as features:\n"
            "  curr     — current window only (columns suffixed _curr)\n"
            "  prev     — previous window only (columns suffixed _prev)\n"
            "  curr+prev — both windows concatenated (default)\n"
        ),
    )
    args = parser.parse_args()

    label_tag    = "neural_labels" if args.filter_labels else "unfiltered"
    model_name   = args.model
    feature_mode = args.feature_mode
    short        = NAMES[model_name]["short"]
    full         = NAMES[model_name]["full"]
    mode_tag     = feature_mode.replace("+", "plus")

    feature_dir_template = f"results/training_data/{label_tag}/hrs{{window}}"

    print_timestamp(
        f"=== Training time-agnostic {full} | features={feature_mode} | "
        f"labels={label_tag} ==="
    )
    print_timestamp(f"Feature directory template: {feature_dir_template}")
    if feature_mode in ("prev", "curr+prev"):
        print_timestamp(
            "Window → predecessor map: "
            + ", ".join(f"{k}→{v}" for k, v in PREV_WINDOW.items())
        )

    # Process tissues in parallel (one worker per tissue)
    results = {}
    with ProcessPoolExecutor(max_workers=len(TISSUES)) as executor:
        futures = {
            executor.submit(
                train_tissue,
                t, feature_dir_template, label_tag,
                model_name, feature_mode, N_SPLITS,
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
            "tissue":       t,
            "model":        model_name,
            "label_tag":    label_tag,
            "feature_mode": feature_mode,
            "mean_auc":     round(r["mean_auc"], 6),
            "std_auc":      round(r["std_auc"],  6),
            "mean_acc":     round(r["mean_acc"], 6),
            "std_acc":      round(r["std_acc"],  6),
        })

    summary_df   = pd.DataFrame(rows)
    summary_dir  = f"results/time_agnostic/{short}/{mode_tag}"
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(
        summary_dir, f"cv_aucroc_summary_{short}_{mode_tag}.csv"
    )
    summary_df.to_csv(summary_path, index=False)
    print_timestamp(f"Summary table saved to {summary_path}")

    print("\n" + summary_df.to_string(index=False))
    print_timestamp("=== All done ===")


if __name__ == "__main__":
    main()