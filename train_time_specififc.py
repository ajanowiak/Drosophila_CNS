import os
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from utils import print_timestamp, make_names_dict


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WINDOWS = ["06-08", "10-12", "14-16"]
TISSUES = ["Neuroblasts", "Neurons", "Glia"]
N_SPLITS = 10

MODEL_CLASSES = {
    "RandomForestClassifier": RandomForestClassifier,
    "SVC":                    SVC,
    "LogisticRegression":     LogisticRegression,
    "XGBClassifier":          XGBClassifier,
}

MODEL_PARAMS = {
    "RandomForestClassifier": dict(n_estimators=500, random_state=0, n_jobs=-1),
    "SVC":                    dict(probability=True),
    "LogisticRegression":     dict(max_iter=1000),
    "XGBClassifier":          dict(n_estimators=500, n_jobs=-1),
}

NAMES = make_names_dict()


# ---------------------------------------------------------------------------
# Core logic (UNCHANGED behavior)
# ---------------------------------------------------------------------------

def run_time_specific(model_name):

    classifier = MODEL_CLASSES[model_name](**MODEL_PARAMS[model_name])

    short = NAMES[model_name]["short"]
    full  = NAMES[model_name]["full"]

    print_timestamp(f"=== Time-specific training: {full} ===")

    for w in WINDOWS:

        print_timestamp(f"--- Window hrs{w} ---")

        X = pd.read_csv(f"data/training/hrs{w}/data_diff_hrs{w}.csv", index_col=0)

        tissue_plotting_info = {}
        summary_rows = []

        for t in TISSUES:

            y_path = f"data/training/hrs{w}/y_{t}.csv"
            if not os.path.exists(y_path):
                continue

            y = pd.read_csv(y_path, index_col=0).iloc[:, 0]
            X_t = X.loc[y.index]

            kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=0)

            probs, trues, accs = [], [], []
            fold_index = {}

            for i, (train_idx, test_idx) in enumerate(kf.split(X_t)):
                X_train, X_test = X_t.iloc[train_idx], X_t.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                fold_index[i] = (train_idx, test_idx)

                clf = clone(classifier)
                clf.fit(X_train, y_train)

                p = clf.predict_proba(X_test)[:, 1]

                probs.append(p)
                trues.append(y_test.values)
                accs.append(accuracy_score(y_test, (p > 0.5).astype(int)))

            # ---- Metrics ----
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

            for i in range(N_SPLITS):
                interp = np.interp(mean_fpr, fprs[i], tprs[i])
                interp[0] = 0.0
                interp_tprs.append(interp)

            mean_tpr = np.mean(interp_tprs, axis=0)
            mean_tpr[-1] = 1.0

            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc  = np.std(roc_aucs)

            std_tpr    = np.std(interp_tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

            # ---- Plot (per tissue) ----
            fig, ax = plt.subplots(figsize=(6, 6))

            ax.plot(
                mean_fpr,
                mean_tpr,
                label=f"AUC = {mean_auc:.3f} ± {std_auc:.3f}",
                lw=1
            )

            ax.fill_between(mean_fpr, tprs_lower, tprs_upper, alpha=0.2)

            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.grid()

            ax.set(
                xlabel="False Positive Rate",
                ylabel="True Positive Rate",
                title=(
                    f"{full} ROC - {t}, hrs{w}\n"
                    f"AUC = {mean_auc:.3f} ± {std_auc:.3f}\n"
                    f"Acc = {mean_acc:.3f} ± {std_acc:.3f}"
                )
            )

            ax.legend(loc="lower right")

            fig_dir = f"results/figures/time_specific/{short}/hrs{w}"
            os.makedirs(fig_dir, exist_ok=True)

            for fmt in ("png", "pdf"):
                fig.savefig(
                    f"{fig_dir}/roc_{short}_{t}_hrs{w}.{fmt}",
                    dpi=300,
                    bbox_inches="tight"
                )

            plt.close(fig)

            # ---- Save models ----
            best_fold = np.argmax(roc_aucs)
            train_idx, _ = fold_index[best_fold]

            best_clf = clone(classifier)
            best_clf.fit(X_t.iloc[train_idx], y.iloc[train_idx])

            cv_dir = f"results/models/time_specific/cv/{short}/hrs{w}"
            os.makedirs(cv_dir, exist_ok=True)
            pickle.dump(best_clf, open(f"{cv_dir}/{short}_{t}_hrs{w}.pkl", "wb"))

            all_clf = clone(classifier)
            all_clf.fit(X_t, y)

            all_dir = f"results/models/time_specific/all_data/{short}/hrs{w}"
            os.makedirs(all_dir, exist_ok=True)
            pickle.dump(all_clf, open(f"{all_dir}/{short}_{t}_hrs{w}.pkl", "wb"))

            # ---- Store for combined plot ----
            tissue_plotting_info[t] = {
                "mean_fpr": mean_fpr,
                "mean_tpr": mean_tpr,
                "tprs_upper": tprs_upper,
                "tprs_lower": tprs_lower,
                "mean_auc": mean_auc,
                "std_auc": std_auc,
            }

            # ---- CSV row ----
            summary_rows.append({
                "tissue": t,
                "model": model_name,
                "window": w,
                "mean_auc": round(mean_auc, 6),
                "std_auc": round(std_auc, 6),
                "mean_acc": round(mean_acc, 6),
                "std_acc": round(std_acc, 6),
            })

            print_timestamp(f"{t} hrs{w} done")

        # ---- Combined plot ----
        fig, ax = plt.subplots(figsize=(6, 6))

        for t, res in tissue_plotting_info.items():
            ax.plot(res["mean_fpr"], res["mean_tpr"], label=t)
            ax.fill_between(res["mean_fpr"], res["tprs_lower"], res["tprs_upper"], alpha=0.2)

        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set(title=f"{full} ROC curves — hrs{w}")
        ax.legend()

        fig_dir = f"results/figures/time_specific/{short}/hrs{w}"

        for fmt in ("png", "pdf"):
            fig.savefig(
                f"{fig_dir}/roc_{short}_combined_hrs{w}.{fmt}",
                dpi=300,
                bbox_inches="tight"
            )

        plt.close(fig)

        # ---- Save CSV ----
        df = pd.DataFrame(summary_rows)

        summary_dir = f"results/time_specific/{short}/hrs{w}"
        os.makedirs(summary_dir, exist_ok=True)

        csv_path = f"{summary_dir}/cv_aucroc_summary_{short}_hrs{w}.csv"
        df.to_csv(csv_path, index=False)

        print_timestamp(f"Saved summary: {csv_path}")

    print_timestamp("=== Done ===")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Time-specific classifier training")
    parser.add_argument(
        "--model",
        choices=list(MODEL_CLASSES.keys()),
        default="RandomForestClassifier"
    )
    args = parser.parse_args()

    run_time_specific(args.model)


if __name__ == "__main__":
    main()