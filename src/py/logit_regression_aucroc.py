# logit_regression_aucroc.py

import argparse
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc

#### FIXME: LOGGER!!
from utils import print_timestamp

#### FIXME: funkcję będą w jakimś utils 
# (albo po prostu potrzebny plik będzie wytwarzany i zapisywany przez inny skrypt JEDNORAZOWO)
from regression_coefs import (
    num_features_from_epv,
    downsample_features,
    compose_downsampled_windows,
)

#### FIXME: move to CONFIG/Args
WINDOWS = ["06-08", "10-12", "14-16"]
TISSUES = ["Neuroblasts", "Neurons", "Glia"]
N_SPLITS = 10
P_THRESHOLD = 0.05


def train_logit_cv(
    tissue: str,
    epv: int,
    num_features: int,
    n_splits: int = N_SPLITS,
) -> dict:
    """
    Cross-validated Logit training for one tissue using downsampled features.
    
    Returns a dict with metrics: mean_auc, std_auc, num_features_significant
    """
    print_timestamp(f"  [{tissue}] EPV={epv}, num_features={num_features}")
    
    downsampled_features = downsample_features(tissue, num_features)
    X, y, composite = compose_downsampled_windows(tissue, downsampled_features)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    
    roc_aucs = []
    all_significant_counts = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, composite)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Fit model on training set
        X_train_sm = sm.add_constant(X_train)
        model_train = sm.Logit(y_train, X_train_sm)
        res_train = model_train.fit(disp=False)
        
        # Predict on test set
        X_test_sm = sm.add_constant(X_test)
        probs = res_train.predict(X_test_sm)
        
        # Calculate AUC
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        roc_aucs.append(roc_auc)
        
        # Count significant features (p < 0.05)
        # Exclude intercept (index 0)
        # p_values = res_train.pvalues[1:]
        # n_significant = (p_values < 0.05).sum()
        # all_significant_counts.append(n_significant)
    
    # Calculate final model on full data for significant features count
    # X_full_sm = sm.add_constant(X)
    # model_full = sm.Logit(y, X_full_sm)
    # res_full = model_full.fit(disp=False)
    
    # # Count significant features in final model
    # p_values_full = res_full.pvalues  # Exclude intercept
    # Run regression_coefs.py to obtain the table:
    data_path = f"results/regression_coefs/epv_{epv}"
    regression_summary = pd.read_csv(os.path.join(data_path, f"{tissue}_summary.csv"))

    num_significant_unadjusted = (regression_summary["p_unadjusted"] < P_THRESHOLD).sum()
    num_significant_bh = (regression_summary["p_adjusted_bh"] < P_THRESHOLD).sum()
    num_significant_tsbh = (regression_summary["p_adjusted_tsbh"] < P_THRESHOLD).sum()


    mean_auc = np.mean(roc_aucs)
    std_auc = np.std(roc_aucs)
    
    print_timestamp(
        f"  [{tissue}] AUC={mean_auc:.4f} ± {std_auc:.4f}, "
        f"significant_features={num_significant_bh} (Benjamini-Hochberg)"
    )
    
    return {
        "tissue": tissue,
        "epv": epv,
        "num_features": num_features,
        "num_features_significant_bh": num_significant_bh,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train cross-validated Logit models on EPV-downsampled features"
    )
    parser.add_argument(
        "--epv_values",
        type=int,
        nargs="+",
        default=[2, 5, 10, 15, 20],
        help="EPV (events per variable) values to test (default: 2 5 10 15 20)",
    )
    args = parser.parse_args()
    
    print_timestamp(f"=== Training Logit with CV | EPV values: {args.epv_values} ===")
    
    #### FIXME: MOVE OUTER LOOP TO SNAKEMAKE !!!
    for tissue in TISSUES:
        all_results = []
        print_timestamp(f"\nProcessing tissue: {tissue}")
        
        for epv in args.epv_values:
            try:
                num_features = num_features_from_epv(tissue, epv)
                
                result = train_logit_cv(
                    tissue=tissue,
                    epv=epv,
                    num_features=num_features,
                    n_splits=N_SPLITS,
                )
                all_results.append(result)
                
            except Exception as e:
                print_timestamp(f"  [{tissue}] EPV={epv} FAILED: {e}")
                continue
        

        if not all_results:
            print_timestamp(f"No successful models for tissue: {tissue}")
            continue    
        # Convert to DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Round numeric columns
        results_df["mean_auc"] = results_df["mean_auc"].round(6)
        results_df["std_auc"] = results_df["std_auc"].round(6)
        
        results_df.sort_values("mean_auc", ascending=False, inplace=True)
        # Select and reorder columns
        results_df = results_df[
            ["epv", "num_features", "num_features_significant_bh", "mean_auc", "std_auc"]
        ]
        
        # Save results
        output_dir = "results/logit_regression_cv_aucroc"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{tissue}_logit_cv_results.csv")
        results_df.to_csv(output_path, index=False)
        
        print_timestamp(f"\nTable for tissue {tissue} saved at {output_path} ")
        # print("\n" + results_df.to_string(index=False))


if __name__ == "__main__":
    main()