# shap_analysis.py

import shap

import os
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import make_names_dict, compose_windows

#### FIXME: (somehow) Add logger


def shap_analysis_with_beeswarm(
    classifier_path: str,
    tissue: str,
    motif_annotations_path: str = None,
    motif_annotations_sep: str = None,
    windows: list[str] = ["06-08", "10-12", "14-16"],
    top_n_motifs: int = 20,
    random_state:int = 0
):
    """
    Perform SHAP-based interpretability analysis for a pretrained
    time-agnostic chromatin loop classifier.

    This function:
    1. Loads a pretrained classifier from disk (.pkl)
    2. Reconstructs the time-agnostic dataset for a given tissue
    3. Computes SHAP values using an explainer appropriate to the model class
    4. Produces and saves a SHAP beeswarm plot
    5. Constructs and saves a dataframe containing SHAP statistics for ALL motifs:
       - mean SHAP value
       - absolute mean SHAP value
       - standard deviation of SHAP values

    Args:
        classifier_path (str): Path to a trained time-agnostic classifier (.pkl file)

        tissue (str): Tissue name (must match label file naming convention)

        motif_annotations_path (str) (Optional): path of the csv file with motif annotations

        motif_annotations_sep (str) (Optional): the separator in said annotations file

        windows (list[str]): Time windows used to construct the time-agnostic dataset

        top_n_motifs (int): Number of motifs to show in the bar plot

        random_state (int): Random seed for reproducibility

    Returns:    
        shap_df (pandas.DataFrame): DataFrame with SHAP statistics for all motifs
    """

    # Load model
    with open(classifier_path, "rb") as f:
        model = pickle.load(f)

    # Create name variables for plot titles/labels and file names
    #### FIXME: (somehow) Load this from config instead!!!!
    model_names = make_names_dict()
    model_key = type(model).__name__
    model_name = model_names[model_key]['full']
    model_short = model_names[model_key]['short']

    #### FIXME: Load the composed .csv into a dataframe instead  
    # Load data
    X, y, composite = compose_windows(tissue, windows)

    ### TUTAJ ROBIĘ X_shap ze zmienioną kolumną !!!
    if motif_annotations_path:
        annot = pd.read_csv(motif_annotations_path, sep=motif_annotations_sep)

        num_mismatches = len(X.columns) - (X.columns == annot["id"]).sum()
        assert num_mismatches == 0, f"Invalid annotations! There are {num_mismatches} mismatches in motif codes!"
        
        new_columns = annot["name"].astype(str) + "  -  (" + annot["id"].astype(str) + ")"
        motif_names = annot["name"]
        motif_ids = annot["id"]
        X.columns = new_columns

    X_shap = X#[:15]    #SMALLER SAMPLE FOR QUICK TESTING

    def predict_proba_pos(model, X):
        """Return P(y=1) only."""
        return model.predict_proba(X)[:, 1]
        
    if model_key in ["RandomForestClassifier", "XGBClassifier"]:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap)

        # Normalize TreeExplainer output
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

    else:
        # Explain only P(y=1)!
        explainer = shap.KernelExplainer(
            lambda X: predict_proba_pos(model, X),
            X_shap
        )
        shap_values = explainer.shap_values(X_shap)

    # Beeswarm plot
    plt.figure(figsize=(8, 8))
    shap.summary_plot(shap_values, X_shap, show=False)

    plt.title(f"{model_name} beeswarm SHAP feature importance plot for tissue {tissue}")
    
    plt.tight_layout()

    # SAVE THE PLOT
    figures_path = f"results/figures/shap/{model_short}"
    os.makedirs(figures_path, exist_ok=True)
    beeswarm_path = os.path.join(
        figures_path, f"{tissue}_beeswarm_{model_short}.pdf"
    )
    plt.savefig(beeswarm_path, dpi=300, format="pdf")

    # Aggregate statistics, create and save DataFrame
    mean_vals = shap_values.mean(axis=0)
    std_vals = shap_values.std(axis=0)
    abs_mean_vals = np.abs(mean_vals)
    mean_abs_vals = np.abs(shap_values).mean(axis=0)

    shap_df = pd.DataFrame({
        "motif_id": motif_ids,
        "motif_name": motif_names,
        "mean_importance": mean_vals,
        "abs_mean_importance": abs_mean_vals,
        "mean_abs_importance": mean_abs_vals,
        "std_importance": std_vals
    }).sort_values("abs_mean_importance", ascending=False)

    # SAVE THE RESULTS DATAFRAME
    data_path = f"results/shap/{model_short}"
    os.makedirs(data_path, exist_ok=True)
    df_path = os.path.join(
        data_path, f"{tissue}_shap_table_{model_short}.csv"
    )
    shap_df.to_csv(df_path, index=False)
    
    return shap_df

def main():

    #### FIXME: MOVE TO CONFIG!!!
    
    tissues = ["Neuroblasts", "Neurons", "Glia"]
    # models = ['RF', 'XGB', 'LR', 'SVM'] 
    models = ['RF', 'XGB'] 
    annot_path = "data/motif_names.tsv"

    #### FIXME: MOVE LOOP TO SNAKEMAKE!!!
    for m in models:
        for t in tissues: 
            print(f"SHAP analysis for model {m} for tissue {t}...")
            _ = shap_analysis_with_beeswarm(
                classifier_path=f"results/models/time_agnostic/all_data/{m}_{t}.pkl",
                tissue=t,
                top_n_motifs=25,
                motif_annotations_path = annot_path,
                motif_annotations_sep = "\t"    
            )

if __name__ == "__main__":
    main()