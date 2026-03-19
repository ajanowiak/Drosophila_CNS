# train_time_agnostic_with_filtering.py

# Trains time-agnostic Random Forest classifiers using per-annotation motif enrichment
# matrices as features. Loops over all refined annotations.
#
# For each annotation:
#   - Features: motif enrichment matrix computed per annotation
#   - Filter loops with < threshold "1-1" cells
#   - Missing loops (rows with any NaN) are dropped before training
#   - Train classifier for each tissue (Neuroblasts, Neurons, Glia)
#   - Only Random Forest, only time-agnostic models
#   - Tissues are processed in parallel (ProcessPoolExecutor)
#   - Records metrics + n_loops per window to a summary CSV per annotation
#
# Usage:
#   python3 train_time_agnostic_with_filtering.py --cells_threshold 1000

import os
import argparse
import pickle
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, accuracy_score

from utils import print_timestamp


WINDOWS   = ["06-08", "10-12", "14-16"]
TISSUES   = ["Neuroblasts", "Neurons", "Glia"]
N_SPLITS  = 10
RF_PARAMS = dict(n_estimators=500, random_state=0, n_jobs=-1)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def get_annotations_from_directory(label_tag: str = "refined_annotations") -> list[str]:
    """
    Scan the results directory to get list of annotations.
    Looks for {annotation}_motif_enrichment_hrs{w}.csv files.
    Returns list of annotation names.
    """
    sample_dir = f"results/training_data/{label_tag}/hrs06-08"
    
    if not os.path.exists(sample_dir):
        return []
    
    annotations = []
    for fname in os.listdir(sample_dir):
        if "_motif_enrichment_hrs" in fname and fname.endswith(".csv"):
            # Extract annotation name from {annotation}_motif_enrichment_hrs{w}.csv
            annotation = fname.replace("_motif_enrichment_hrs06-08.csv", "")
            if annotation and annotation not in annotations:
                annotations.append(annotation)
    return sorted(annotations)


def filter_loops_by_threshold(
    enrichment_df: pd.DataFrame,
    count_df: pd.DataFrame,
    threshold: int
) -> tuple[pd.DataFrame, int]:
    """
    Filter out loops with less than threshold "1-1" cells.
    
    Args:
        enrichment_df: enrichment matrix (n_loops, n_motifs)
        count_df: count matrix or series (n_loops,) or (n_loops, n_tissues)
        threshold: minimum count threshold
    
    Returns:
        filtered enrichment DataFrame and count of remaining loops
    """
    # If count_df is 2D (multiple tissues), take max across tissues
    if len(count_df.shape) > 1:
        counts = count_df.max(axis=1)
    else:
        counts = count_df
    
    # Align indices
    shared_loops = enrichment_df.index.intersection(counts.index)
    enrichment_df = enrichment_df.loc[shared_loops]
    counts = counts.loc[shared_loops]
    
    # Filter by threshold
    mask = counts >= threshold
    enrichment_filtered = enrichment_df[mask]
    n_remaining = len(enrichment_filtered)
    
    return enrichment_filtered, n_remaining


def compose_windows_enrichment(
    tissue: str,
    annotation: str,
    cells_threshold: int,
    label_tag: str = "refined_annotations"
) -> tuple:
    """
    Concatenate enrichment matrices across windows for a given annotation,
    align with tissue labels, filter by cell count threshold,
    drop NaN rows, and build a composite stratification vector.

    Args:
        tissue: tissue label, e.g. "Neuroblasts"
        annotation: annotation label, e.g. "Brain"
        cells_threshold: minimum count of "1-1" cells required for a loop
        label_tag: directory tag for training data

    Returns:
        X (pd.DataFrame), y (pd.Series), composite (np.ndarray of codes), n_loops dict
    """
    Xs, ys, n_loops_dict = [], [], {}

    for idx, w in enumerate(WINDOWS):
        feature_dir = f"results/training_data/{label_tag}/hrs{w}"
        enrich_path = os.path.join(feature_dir, f"{annotation}_motif_enrichment_hrs{w}.csv")
        count_path = os.path.join(feature_dir, f"count11_all_tissues_hrs{w}.csv")
        y_path = os.path.join(feature_dir, f"y_{tissue}.csv")

        if not os.path.exists(enrich_path):
            raise FileNotFoundError(f"Enrichment file not found: {enrich_path}")
        if not os.path.exists(count_path):
            raise FileNotFoundError(f"Count file not found: {count_path}")
        if not os.path.exists(y_path):
            raise FileNotFoundError(f"Label file not found: {y_path}")

        X_w = pd.read_csv(enrich_path, index_col=0)
        count_w = pd.read_csv(count_path, index_col=0)
        y_w = pd.read_csv(y_path, index_col=0).iloc[:, 0]

        # Align on shared loops
        shared = X_w.index.intersection(y_w.index)
        X_w = X_w.loc[shared]
        count_w = count_w.loc[shared]
        y_w = y_w.loc[shared]

        # Filter by cell count threshold
        X_w_filt, n_passing = filter_loops_by_threshold(X_w, count_w, cells_threshold)
        n_loops_dict[w] = n_passing

        # Align label with filtered loops
        y_w = y_w.loc[X_w_filt.index]

        # Drop loops (rows) that have any NaN feature
        before = len(X_w_filt)
        X_w_filt = X_w_filt.dropna(axis=0)
        after = len(X_w_filt)
        if before != after:
            print_timestamp(
                f"  [{tissue}] [annotation={annotation}] hrs{w}: "
                f"dropped {before - after} loops with NaN features ({after} remaining)"
            )
        y_w = y_w.loc[X_w_filt.index]

        # Tag with window index for stratification
        X_w_filt["_window"] = idx
        Xs.append(X_w_filt)
        ys.append(y_w)

    X = pd.concat(Xs, axis=0)
    y = pd.concat(ys, axis=0)

    composite = pd.Categorical(list(zip(X["_window"], y))).codes
    X = X.drop(columns=["_window"])

    return X, y, composite, n_loops_dict


# ---------------------------------------------------------------------------
# Training for a single tissue
# ---------------------------------------------------------------------------

def train_tissue(
    tissue: str,
    annotation: str,
    cells_threshold: int,
    label_tag: str = "refined_annotations",
    n_splits: int = N_SPLITS,
    params: dict = RF_PARAMS,
) -> dict:
    """
    Cross-validated time-agnostic RF training for one tissue using a specific annotation's enrichment.

    Returns a dict with tissue name, annotation, metrics, and n_loops per window.
    """
    print_timestamp(f"[{annotation}] [{tissue}] Starting training...")

    try:
        X, y, composite, n_loops_dict = compose_windows_enrichment(
            tissue, annotation, cells_threshold, label_tag
        )
    except FileNotFoundError as e:
        print_timestamp(f"[{annotation}] [{tissue}] SKIPPED: {e}")
        return None

    print_timestamp(
        f"[{annotation}] [{tissue}] Data shape after threshold+NaN drop: {X.shape}, "
        f"positives: {y.sum()}"
    )

    classifier = RandomForestClassifier(**params)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

    probs, trues, accs = [], [], []

    for i, (train_idx, test_idx) in enumerate(skf.split(X, composite)):
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
    std_acc = np.std(accs)

    fprs, tprs, roc_aucs = [], [], []
    for true, prob in zip(trues, probs):
        fpr, tpr, _ = roc_curve(true, prob)
        fprs.append(fpr)
        tprs.append(tpr)
        roc_aucs.append(auc(fpr, tpr))

    mean_auc = np.mean(roc_aucs)
    std_auc = np.std(roc_aucs)

    # --- Save all-data model ---
    all_clf = clone(classifier)
    all_clf.fit(X, y)

    all_dir = f"results/models/time_agnostic_with_filtering/{annotation}"
    os.makedirs(all_dir, exist_ok=True)
    all_path = os.path.join(all_dir, f"RF_{tissue}.pkl")
    pickle.dump(all_clf, open(all_path, "wb"))

    print_timestamp(
        f"[{annotation}] [{tissue}] Done - mean AUC={mean_auc:.4f} ± {std_auc:.4f}, "
        f"mean Acc={mean_acc:.4f} ± {std_acc:.4f}"
    )

    return {
        "tissue": tissue,
        "annotation": annotation,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "n_loops_06-08": n_loops_dict["06-08"],
        "n_loops_10-12": n_loops_dict["10-12"],
        "n_loops_14-16": n_loops_dict["14-16"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train time-agnostic Random Forest on per-annotation motif enrichment features."
    )
    parser.add_argument(
        "--cells_threshold",
        type=int,
        default=1000,
        help="Minimal number of 1-1 cells that a loop should have in order to be included in training",
    )
    args = parser.parse_args()

    label_tag = "refined_annotations"

    print_timestamp(f"=== Training time-agnostic RF | cells_threshold={args.cells_threshold} ===")
    
    # Get list of annotations by scanning the directory
    annotations = get_annotations_from_directory(label_tag)
    if not annotations:
        print_timestamp("ERROR: No annotation directories found. Make sure generate_motif_enrichment_with_filtering.py has been run.")
        return

    print_timestamp(f"Found {len(annotations)} annotations to train on")

    # For each annotation, train classifiers for all tissues
    all_results = {}
    
    for annotation in annotations:
        print_timestamp(f"\n=== Processing annotation: {annotation} ===")
        
        results = {}
        with ProcessPoolExecutor(max_workers=len(TISSUES)) as executor:
            futures = {
                executor.submit(
                    train_tissue, t, annotation, args.cells_threshold, label_tag, N_SPLITS
                ): t
                for t in TISSUES
            }
            for fut in as_completed(futures):
                t = futures[fut]
                try:
                    result = fut.result()
                    if result is not None:
                        results[t] = result
                except Exception as e:
                    print_timestamp(f"[{annotation}] [{t}] FAILED: {e}")
                    raise

        # --- Summary table per annotation ---
        rows = []
        for t in TISSUES:
            if t not in results:
                continue
            r = results[t]
            rows.append({
                "tissue": r["tissue"],
                "mean_auc": round(r["mean_auc"], 6),
                "std_auc": round(r["std_auc"], 6),
                "mean_acc": round(r["mean_acc"], 6),
                "std_acc": round(r["std_acc"], 6),
                "n_loops_06-08": r["n_loops_06-08"],
                "n_loops_10-12": r["n_loops_10-12"],
                "n_loops_14-16": r["n_loops_14-16"],
            })

        summary_df = pd.DataFrame(rows)
        summary_dir = f"results/time_agnostic_with_filtering"
        os.makedirs(summary_dir, exist_ok=True)
        summary_path = os.path.join(summary_dir, f"{annotation}_cv_aucroc_summary_RF_threshold{args.cells_threshold}.csv")
        summary_df.to_csv(summary_path, index=False)
        print_timestamp(f"Summary table for {annotation} saved to {summary_path}")

        print("\n" + summary_df.to_string(index=False))
        
        all_results[annotation] = summary_df

    print_timestamp("\n=== All annotations done ===")


if __name__ == "__main__":
    main()