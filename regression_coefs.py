import argparse
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import subprocess

from utils import print_timestamp
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

def num_features_from_epv(tissue: str, epv: int = 10) -> int:
    """
    Compute the number of events and the optimal number of features to obtain the desired EPV

    events per variable:
    EPV = number_of_events / number_of_features 
    (events == positive class)
    """

    training_data_dir = "results/training_data/unfiltered/"
    
    windows = ["06-08", "10-12", "14-16"] # paths formatting
    timepoints = ["6-8h", "10-12h", "14-16h"] # columns fomratting


    y_vector = []
    for w, tp in zip(windows, timepoints):
        curr_path = os.path.join(training_data_dir, f"hrs{w}/y_{tissue}.csv")
        curr_df = pd.read_csv(curr_path)
        
        y_vector.extend(curr_df[f"Dmel_{tp}_{tissue}"].to_list())

    num_events = sum(y_vector)

    return int(np.ceil(num_events / epv))

def downsample_features_shap(tissue: str, num_features = None) -> pd.Series:
    shap_importance = pd.read_csv(f"results/shap/XGB/{tissue}_shap_table_XGB.csv")

    # Sortuję po abs_mean_importance
    # Sortowanie po mean_abs_importance daje wyraźnie inne wyniki, ale bardzo podobne 
    # do takich jakbyśmy dostali biorąc motywy o największym odchyleniu standardowym importance
    
    sorted_features = shap_importance.sort_values('abs_mean_importance', ascending=False)["motif_id"]
    
    if num_features:
        return sorted_features[:num_features]
    else:
        return sorted_features

def downsample_features(tissue: str, num_features = None) -> pd.Series:
    # Simpler version (relative to XGBoost and SHAP): features are downsampled based on Mean Decrease in Impurity in time-agnostic Random Forest model.

    # importance = pd.read_csv(f"results/RF_MDI_importance/{tissue}_importance_table.csv")
    # sorted_features = importance.sort_values('mean_importance', ascending=False)["motif_id"]
    
    importance = pd.read_csv(f"results/RF_permutation_importance/{tissue}_permutation_importance_table.csv")
    sorted_features = importance.sort_values('abs_mean_importance', ascending=False)["motif_id"]
    
    if num_features:
        return sorted_features[:num_features]
    else:
        return sorted_features
    
def compose_downsampled_windows(tissue: str, downsampled_features: pd.Series, windows:list[str] = ["06-08", "10-12", "14-16"]) -> tuple:
    """
    TIME-AGNOSTIC CLASSIFIER
    concatenate window-specific DataFrames and generate a `composite` vector for stratification
    
    Extract top n features based on a previous shap analysis
    """
    Xs, ys = [], []
    training_dir = "results/training_data/unfiltered"
    
    for idx, w in enumerate(windows):

        curr_X = pd.read_csv(os.path.join(training_dir, f"hrs{w}/motif_enrichment_hrs{w}.csv"), index_col=0)
        curr_y = pd.read_csv(os.path.join(training_dir, f"hrs{w}/y_{tissue}.csv"), index_col=0).iloc[:, 0]
        
        curr_X = curr_X.dropna(axis=0)
        curr_X = curr_X[downsampled_features] # crucial change
        
        curr_X['window'] = idx

        # align X and y
        common_index = curr_X.index.intersection(curr_y.index)
        curr_X = curr_X.loc[common_index]
        curr_y = curr_y.loc[common_index]

        Xs.append(curr_X)
        ys.append(curr_y)

    y_new = pd.concat(ys, axis=0)
    X_new = pd.concat(Xs, axis=0)

    composite = pd.Categorical(list(zip(X_new['window'], y_new))).codes

    X_new.drop('window', axis=1, inplace=True) # we don't want to use 'window' for prediction

    return X_new, y_new, composite

def plot_coeffs(summary_df, tissue, num_features, output_path=None, top_n=20):
    """
    Plot logistic regression coefficients with 95% confidence intervals.
    
    Args:
        summary_df: DataFrame with regression results
        tissue: Tissue name for title
        num_features: Number of features used
        output_path: If provided, save plot to this path
        top_n: Number of top features to display
    """
    df_plot = summary_df.head(top_n).iloc[::-1]

    plt.figure(figsize=(8, 6))

    plt.errorbar(
        df_plot["coef"],
        df_plot.index,
        xerr=[
            df_plot["coef"] - df_plot["ci_lower"],
            df_plot["ci_upper"] - df_plot["coef"]
        ],
        fmt='o'
    )

    plt.axvline(0, linestyle="--", color='r')
    plt.xlabel("Coefficient (log-odds)")
    plt.title(f"Top {top_n} Features with 95% CI - {tissue}")
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, format="pdf")
        plt.close()
    else:
        plt.show()

def plot_volcano(summary_df, tissue, num_features, output_path=None, p_thresh=0.05, effect_thresh=0.5):
    """
    Create volcano plot of logistic regression coefficients.
    
    Args:
        summary_df: DataFrame with regression results
        tissue: Tissue name for title
        num_features: Number of features used
        output_path: If provided, save plot to this path
        p_thresh: P-value threshold for significance
        effect_thresh: Effect size threshold
    """
    df = summary_df.copy()
    p_column_name = "p_adjusted_bh" 

    print(df[p_column_name].describe())
    print((df[p_column_name] == 0).sum())
    print(df[p_column_name].isna().sum())

    # Categories
    conditions = [
        (df[p_column_name] < p_thresh) & (df["coef"] > effect_thresh),
        (df[p_column_name] < p_thresh) & (df["coef"] < -effect_thresh),
        (df[p_column_name] < p_thresh)
    ]

    choices = [
        "positive_strong",
        "negative_strong",
        "significant_only"
    ]

    df["category"] = np.select(conditions, choices, default="other")

    # Plot
    plt.figure(figsize=(7, 6))

    # Plot all points
    for cat, group in df.groupby("category"):
        if cat == "positive_strong":
            plt.scatter(group["coef"], -np.log10(group[p_column_name]), label="Strong positive", c='firebrick')
        elif cat == "negative_strong":
            plt.scatter(group["coef"], -np.log10(group[p_column_name]), label="Strong negative", c='navy')
        elif cat == "significant_only":
            plt.scatter(group["coef"], -np.log10(group[p_column_name]), label="Significant small effect")
        else:
            plt.scatter(group["coef"], -np.log10(group[p_column_name]), alpha=0.3, c='grey')

    # Threshold lines
    plt.axhline(-np.log10(p_thresh), c='black', linestyle='dotted')
    plt.axvline(effect_thresh, c='black', linestyle='dotted')
    plt.axvline(-effect_thresh, c='black', linestyle='dotted')

    plt.xlabel("Effect size (coef)")
    plt.ylabel("-log10(adjusted p-value (Benjamini-Hochberg))")
    plt.title(f"Volcano Plot - {tissue} ({num_features} features)\nP-value threshold: {p_thresh}")

    plt.legend()
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, format="pdf")
        plt.close()
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Perform logistic regression analysis on motif enrichment data")
    parser.add_argument("--epv", type=int, default=10, help="Events per variable for feature selection (default: 10)")
    args = parser.parse_args()
    
    windows = ["06-08", "10-12", "14-16"]
    tissues = ["Neuroblasts", "Neurons", "Glia"]
    model_name = "Logistic Regression"
    model_short = "LR"

    epv = args.epv
    motif_annotations_path = "data/motif_names.tsv"
    top_n = 20

    for tissue in tissues:
        print_timestamp(f"Processing tissue: {tissue}")
        
        num_features = num_features_from_epv(tissue, epv)
        print_timestamp(f"Selected {num_features} features for tissue {tissue} and EPV = {epv}")

        # one of two functions here: downsample_features() or downsample_features_shap()
        downsampled_features = downsample_features(tissue, num_features)

        X, y, composite = compose_downsampled_windows(tissue, downsampled_features)

        # Load motif annotations
        annot = pd.read_csv(motif_annotations_path, sep="\t")

        code_to_name = (
            annot
            .drop_duplicates("id")
            .set_index("id")["name"]
            .astype(str)
            .to_dict()
        )

        for name, id in code_to_name.items():
            code_to_name[name] = str(name) + "  -  (" + str(id) + ")"

        mapped_columns = (
            X.columns
            .to_series()
            .map(code_to_name)
            .fillna(pd.Series(X.columns, index=X.columns))
        )

        X_named = X.copy()
        X_named.columns = mapped_columns

        # Final fit on full data
        X_full_sm = sm.add_constant(X_named)
        model_full = sm.Logit(y, X_full_sm)

        res_full = model_full.fit(disp=False)

        p_vals_unadjusted = res_full.pvalues
        _, p_vals_bh, _, _ = multipletests(p_vals_unadjusted, method="fdr_bh")  # Benjamini-Hochberg
        _, p_vals_tsbh, _, _ = multipletests(p_vals_unadjusted, method="fdr_tsbh") # two stage Benjamini-Hochberg (less conservative (?))

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
        figures_path = f"results/figures/regression_coefs/epv_{epv}"
        data_path = f"results/regression_coefs/epv_{epv}"
        os.makedirs(figures_path, exist_ok=True)
        os.makedirs(data_path, exist_ok=True)

        # Save plots
        print_timestamp(f"Saving plots for tissue {tissue}...")
        
        coef_plot_path = os.path.join(figures_path, f"{tissue}_coefficients.pdf")
        plot_coeffs(summary_df, tissue, num_features, output_path=coef_plot_path, top_n=top_n)

        volcano_plot_path = os.path.join(figures_path, f"{tissue}_volcano.pdf")
        plot_volcano(summary_df, tissue, num_features, output_path=volcano_plot_path)

        # Save summary table
        print_timestamp(f"Saving summary table for tissue {tissue}...")
        
        summary_table_path = os.path.join(data_path, f"{tissue}_summary.csv")
        summary_df.to_csv(summary_table_path)

        print_timestamp(f"Completed processing for tissue {tissue}")


if __name__ == "__main__":
    main()