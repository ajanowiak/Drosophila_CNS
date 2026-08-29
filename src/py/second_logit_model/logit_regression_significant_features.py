# logit_regression_significant_features.py

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests

#### FIXME: LOGGER !!
from utils import print_timestamp

#### FIXME: import will come from elsewhere (some utils) OR the downsampled feature file will be prepared elsewhere
from regression_coefs import compose_downsampled_windows, plot_coeffs, plot_volcano

#### FIXME: plot_coeffs and plot_volcano will also be handled differently  (utils function)

#### FIXME: CONFIG/Args
TISSUES = ["Neuroblasts", "Neurons", "Glia"]
N_SPLITS = 10
P_THRESHOLD = 0.05


def extract_significant_features(tissue: str, epv: int = 10) -> pd.Series:
    #### FIXME: motif name formatting will be handled by utils function or lookup table !!
    """
    Extract feature names (motif_ids) with BH-adjusted p-value < 0.05
    from regression_coefs.py output.
    
    Returns a pd.Series of motif IDs (feature names)
    """
    data_path = f"results/regression_coefs/epv_{epv}"
    summary_csv = os.path.join(data_path, f"{tissue}_summary.csv")
    
    if not os.path.exists(summary_csv):
        raise FileNotFoundError(f"Summary file not found: {summary_csv}")
    
    summary_df = pd.read_csv(summary_csv, index_col=0)
    
    # Filter by BH-adjusted p-value
    significant_df = summary_df[summary_df["p_adjusted_bh"] < P_THRESHOLD]
    
    # Extract motif IDs from index
    # The index contains annotated names like "motif_id  -  (annotation)"
    # We need to extract the motif_id part
    motif_ids = []
    for idx in significant_df.index:
        # Index format is like "motif_id  -  (annotation)" or just "motif_id"
        motif_id = idx.split("  -  ")[0].strip()
        motif_ids.append(motif_id)
    
    return pd.Series(motif_ids)


def train_logit_cv_significant(
    tissue: str,
    significant_features: pd.Series,
    epv: int = 10,
    n_splits: int = N_SPLITS,
) -> dict:
    """
    Cross-validated Logit training using only significant features.
    
    Returns a dict with metrics: mean_auc, std_auc, num_features
    """
    num_features = len(significant_features)
    print_timestamp(f"  [{tissue}] Training with {num_features} significant features")
    
    X, y, composite = compose_downsampled_windows(tissue, significant_features)
    
    #### FIXME: DLACZEGO TU JEST CV A W regression_coefs.py NIE MA !?!?!????

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    
    roc_aucs = []
    
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
    
    mean_auc = np.mean(roc_aucs)
    std_auc = np.std(roc_aucs)
    
    print_timestamp(
        f"  [{tissue}] AUC={mean_auc:.4f} ± {std_auc:.4f} ({num_features} significant features)"
    )
    
    return {
        "tissue": tissue,
        "num_features": num_features,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
    }


def main():
    # EPV configuration per tissue
    epv_config = {
        "Neuroblasts": 15,
        "Neurons": 15,
        "Glia": 15,
    }
    
    motif_annotations_path = "data/motif_names.tsv"
    
    print_timestamp(f"=== Training Logit with Significant Features ===")
    
    #### FIXME: move outer loop to snakemake !!!
    for tissue in TISSUES:
        print_timestamp(f"\nProcessing tissue: {tissue}")
        
        epv = epv_config[tissue]
        
        try:
            # Extract significant features
            significant_features = extract_significant_features(tissue, epv)
            
            if len(significant_features) == 0:
                print_timestamp(f"  [{tissue}] No significant features found at p < 0.05")
                continue
            
            print_timestamp(f"  [{tissue}] Found {len(significant_features)} significant features")
            
            # CV training
            result = train_logit_cv_significant(
                tissue=tissue,
                significant_features=significant_features,
                epv=epv,
                n_splits=N_SPLITS,
            )
            
            # Train final model on full data for coefficient estimation
            X, y, composite = compose_downsampled_windows(tissue, significant_features)
            
            # Load motif annotations
            annot = pd.read_csv(motif_annotations_path, sep="\t")
            code_to_name = (
                annot
                .drop_duplicates("id")
                .set_index("id")["name"]
                .astype(str)
                .to_dict()
            )
            
            #### FIXME: this will be handled by utils function or lookup table
            # Create mapping for display
            mapped_columns = pd.Series(index=X.columns, dtype=str)
            for col in X.columns:
                if col in code_to_name:
                    mapped_columns[col] = f"{col}  -  ({code_to_name[col]})"
                else:
                    mapped_columns[col] = col
            
            X_named = X.copy()
            X_named.columns = mapped_columns
            
            # Fit final model
            X_full_sm = sm.add_constant(X_named)
            model_full = sm.Logit(y, X_full_sm)
            res_full = model_full.fit(disp=False)
            
            # Prepare summary with p-values and confidence intervals
            p_vals_unadjusted = res_full.pvalues
            _, p_vals_bh, _, _ = multipletests(p_vals_unadjusted, method="fdr_bh")  # Benjamini-Hochberg
            
            #### FIXME: remove tsbh
            _, p_vals_tsbh, _, _ = multipletests(p_vals_unadjusted, method="fdr_tsbh") # two stage Benjamini-Hochberg
            
            summary_df = pd.DataFrame({
                "coef": res_full.params,
                "std_err": res_full.bse,
                "z": res_full.tvalues,
                "p_unadjusted": p_vals_unadjusted,
                "p_adjusted_bh": p_vals_bh,
                "p_adjusted_tsbh": p_vals_tsbh,
            })
            
            # Confidence intervals
            ci = res_full.conf_int()
            ci.columns = ["ci_lower", "ci_upper"]
            summary_df = pd.concat([summary_df, ci], axis=1)
            
            # Sort by importance
            summary_df = summary_df.sort_values("coef", key=abs, ascending=False)
            
            # Create output directories
            figures_path = f"results/figures/regression_coefs/significant_features_based_on_epv_{epv}"
            data_path = f"results/regression_coefs/significant_features_based_on_epv_{epv}"
            os.makedirs(figures_path, exist_ok=True)
            os.makedirs(data_path, exist_ok=True)
            
            # Save CV results
            cv_results_df = pd.DataFrame([result])
            cv_results_path = os.path.join(data_path, f"{tissue}_cv_results.csv")
            cv_results_df.to_csv(cv_results_path, index=False)
            print_timestamp(f"Saved CV results to {cv_results_path}")
            
            # Save summary table
            print_timestamp(f"Saving summary table for tissue {tissue}...")
            summary_table_path = os.path.join(data_path, f"{tissue}_summary.csv")
            summary_df.to_csv(summary_table_path)
            
            # Save plots
            print_timestamp(f"Saving plots for tissue {tissue}...")
            top_n = 20
            coef_plot_path = os.path.join(figures_path, f"{tissue}_coefficients.pdf")
            plot_coeffs(summary_df, tissue, len(significant_features), output_path=coef_plot_path, top_n=top_n)
            
            volcano_plot_path = os.path.join(figures_path, f"{tissue}_volcano.pdf")
            plot_volcano(summary_df, tissue, len(significant_features), output_path=volcano_plot_path)
            
        except Exception as e:
            print_timestamp(f"  [{tissue}] FAILED: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print_timestamp("\n=== Completed ===")


if __name__ == "__main__":
    main()
