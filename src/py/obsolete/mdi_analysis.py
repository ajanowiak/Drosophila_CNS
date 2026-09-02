# mdi_analysis.py
# MDI = Mean Decrease in Impurity

# Feature importance analysis of time-agnostic Random Forest
# EPV-based feature selection in regression_coefs.py will be based on MDI
# można jeszcze jakieś permutation importance zrobić

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import compose_windows


def rf_importance_analysis(
    classifier_path: str,
    tissue: str,
    motif_annotations_path: str = None,
    motif_annotations_sep: str = None,
    windows: list[str] = ["06-08", "10-12", "14-16"],
    top_n_motifs: int = 20
):
    """
    Analyse feature importance scores for a pretrained
    time-agnostic Random Forest classifier.

    This function:
    1. Loads a pretrained RF classifier
    2. Reconstructs the time-agnostic dataset
    3. Extracts feature importance scores
    4. Saves a bar plot of the top motifs
    5. Saves a full importance table

    Returns:
        importance_df (pd.DataFrame)
    """

    # Load model
    with open(classifier_path, "rb") as f:
        model = pickle.load(f)

    model_name = "RandomForest"

    # Load data
    X, y, composite = compose_windows(tissue, windows)

    motif_ids = X.columns
    motif_names = X.columns

    # Replace motif IDs with readable names if annotations are provided
    if motif_annotations_path:
        annot = pd.read_csv(
            motif_annotations_path,
            sep=motif_annotations_sep
        )

        num_mismatches = len(X.columns) - (X.columns == annot["id"]).sum()
        assert num_mismatches == 0, (
            f"Invalid annotations! "
            f"There are {num_mismatches} mismatches in motif codes."
        )

        motif_ids = annot["id"]
        motif_names = annot["name"]

    # Extract RF importance scores from each tree
    tree_importances = [tree.feature_importances_ for tree in model.estimators_]
    
    # Compute mean and std across trees
    mean_importance = np.mean(tree_importances, axis=0)
    std_importance = np.std(tree_importances, axis=0)

    # Create results dataframe
    importance_df = pd.DataFrame({
        "motif_id": motif_ids,
        "motif_name": motif_names,
        "mean_importance": mean_importance,
        "std_importance": std_importance
    }).sort_values("mean_importance", ascending=False)

    # Plot top motifs
    top_df = importance_df.head(top_n_motifs).copy()
    top_df = top_df.iloc[::-1]   # reverse for better top-to-bottom ordering

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.barh(
        top_df["motif_name"],
        top_df["mean_importance"],
        xerr=top_df["std_importance"],
        color = "mediumseagreen",
        alpha = 0.8,
        edgecolor="black",
        linewidth=0.6,
        capsize=4,
        error_kw={"elinewidth": 1, "alpha": 0.7}
    )

    ax.set_xlabel("Mean feature importance across trees", fontsize=11)
    ax.set_ylabel("")
    ax.set_title(
        f"Mean feature importance across trees in the Random Forest model\nTissue: {tissue}. Top {top_n_motifs} features",
        fontsize=13,
        pad=12
    )

    # cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # add subtle x-grid for readability
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    # improve label readability
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=9)

    plt.tight_layout()

    # Save figure
    figures_path = f"results/figures/RF_MDI_importance"
    os.makedirs(figures_path, exist_ok=True)

    fig_path = os.path.join(
        figures_path,
        f"{tissue}_mdi_importance.pdf"
    )
    plt.savefig(fig_path, dpi=300, format="pdf")

    # Save full table
    data_path = f"results/RF_MDI_importance"
    os.makedirs(data_path, exist_ok=True)

    df_path = os.path.join(
        data_path,
        f"{tissue}_mdi_importance_table.csv"
    )
    importance_df.to_csv(df_path, index=False)

    return importance_df


def main():
    tissues = ["Neuroblasts", "Neurons", "Glia"]
    annot_path = "data/motif_names.tsv"

    for tissue in tissues:
        print(f"RF importance analysis for tissue {tissue}...")

        _ = rf_importance_analysis(
            classifier_path=f"results/models/time_agnostic/all_data/RF_{tissue}.pkl",
            tissue=tissue,
            top_n_motifs=35,
            motif_annotations_path=annot_path,
            motif_annotations_sep="\t"
        )


if __name__ == "__main__":
    main()