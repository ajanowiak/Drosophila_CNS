# compute_enrichment_difference.py
#
# For each time window:
#   1. Loads enrichment tables
#   2. Adds binary columns for loop presence (by tissue)
#   3. Drops missing loops (rows with NaN values)
#   4. Creates and saves motif importance table from SHAP values
#
# Output:
#   1. Enrichment tables with added loop presence columns
#   2. New motif importance table (motifs x mean_shap_importance columns)

import os
import numpy as np
import pandas as pd

from utils import print_timestamp


WINDOWS = ['06-08', '10-12', '14-16']
TISSUES = ['Neuroblasts', 'Neurons', 'Glia']
MODELS = ['RF', 'XGB']


def load_loop_presence(window: str) -> pd.DataFrame:
    """
    Load loop presence information from the TSV file.
    Returns a DataFrame with loop_ids as index and tissue-specific binary columns.
    """
    loop_file = "data/long_and_short_range_loops_D_mel.tsv"
    loop_df = pd.read_csv(loop_file, sep='\t', index_col='loop_id')
    
    # Map windows to TSV column prefixes
    window_map = {
        '06-08': 'Dmel_6-8h',
        '10-12': 'Dmel_10-12h',
        '14-16': 'Dmel_14-16h'
    }
    prefix = window_map[window]
    
    if not prefix:
        raise ValueError(f"Unknown window: {window}")
    
    # Extract tissue columns for this window
    tissue_columns = {
        tissue: f"{prefix}_{tissue}" for tissue in TISSUES
    }
    
    # Select only the relevant columns
    selected_cols = [col for col in tissue_columns.values() if col in loop_df.columns]
    tissue_data = loop_df[selected_cols].copy()
    
    # Rename to simple tissue names
    tissue_data.columns = ["presence_in_" + col.split('_')[-1] for col in tissue_data.columns]
    
    return tissue_data


def add_loop_presence_columns(enrichment_df: pd.DataFrame, loop_presence_df: pd.DataFrame) -> tuple:
    """
    Add binary loop presence columns to enrichment table.
    Aligns loops by index.
    Returns (enrichment_df_with_presence, shared_loops)
    """
    # Keep only loops that exist in both tables
    shared_loops = enrichment_df.index.intersection(loop_presence_df.index)
    enrichment_aligned = enrichment_df.loc[shared_loops].copy()
    loop_presence_aligned = loop_presence_df.loc[shared_loops].copy()
    
    # Concatenate the presence columns (put tissue columns first)
    result_df = pd.concat([loop_presence_aligned, enrichment_aligned], axis=1)
    
    return result_df, shared_loops


def drop_missing_loops(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows (loops) with NaN values.
    Returns (cleaned_df, original_shape, new_shape)
    """
    original_shape = df.shape
    result = df.dropna()
    new_shape = result.shape
    print_timestamp(f"Dropped missing loops: {original_shape} -> {new_shape}")
    
    return result


def load_shap_tables() -> pd.DataFrame:
    """
    Load all SHAP importance tables and combine them.
    Returns a DataFrame with motif_id as index and columns:
    mean_shap_importance_{model}_{tissue}
    """
    print_timestamp("Loading SHAP importance tables...")
    
    shap_data = {}
    
    for model in MODELS:
        for tissue in TISSUES:
            file_path = f"results/shap/{model}/{tissue}_shap_table_{model}.csv"
            
            if not os.path.exists(file_path):
                continue
            
            try:
                shap_df = pd.read_csv(file_path, index_col='motif_id')
                
                # Use mean_importance column
                col_name = f"mean_shap_importance_{model}_{tissue}"
                shap_data[col_name] = shap_df['mean_importance']
            except Exception as e:
                print_timestamp(f"Error reading {file_path}: {e}")
                continue
    
    if not shap_data:
        raise ValueError("No SHAP tables could be loaded. Check file paths.")
    
    # Combine all SHAP data into one DataFrame
    combined_shap = pd.concat(shap_data, axis=1)
    
    return combined_shap


def create_motif_importance_table(window: str, motif_ids: list, motif_enrichment: pd.Series) -> pd.DataFrame:
    """
    Create a table with motifs as rows and mean_shap_importance columns + mean enrichment.
    Only includes motifs that exist in the given list.
    motif_enrichment: Series with motif_id as index and mean enrichment values
    """
    # Load all SHAP data
    shap_combined = load_shap_tables()
    
    # Filter SHAP data to only include motifs present in this window
    motif_importance_table = shap_combined.loc[shap_combined.index.isin(motif_ids)].copy()

    # Add mean enrichment across loops
    motif_importance_table.insert(
        0,
        "mean_enrichment_difference_across_loops",
        motif_enrichment
    )   
    return motif_importance_table


def compute_difference_for_window(window: str) -> None:
    neural_dir      = f"results/training_data/neural_labels/hrs{window}"
    unfiltered_dir  = f"results/training_data/unfiltered/hrs{window}"
    output_dir      = f"results/EDA/enrichment_difference"
    os.makedirs(output_dir, exist_ok=True)

    # --- load enrichment tables ---
    neural_path     = os.path.join(neural_dir,     f"motif_enrichment_hrs{window}.csv")
    unfiltered_path = os.path.join(unfiltered_dir, f"motif_enrichment_hrs{window}.csv")

    print_timestamp(f"Window {window}: loading enrichment tables...")
    neural_df     = pd.read_csv(neural_path,     index_col=0)
    unfiltered_df = pd.read_csv(unfiltered_path, index_col=0)

    # --- load count_11 (take from neural; value is loop-level so either file works) ---
    count_path = os.path.join(neural_dir, f"count11_hrs{window}.csv")
    print_timestamp(f"Window {window}: loading count_11 table...")
    count_df = pd.read_csv(count_path, index_col=0)
    # All motif columns hold the same value per row — take the first column
    count_11 = count_df.iloc[:, 0].rename("count_11")

    # --- align indices / columns (defensive) ---
    shared_loops  = neural_df.index.intersection(unfiltered_df.index)
    shared_motifs = neural_df.columns.intersection(unfiltered_df.columns)

    neural_df     = neural_df.loc[shared_loops, shared_motifs]
    unfiltered_df = unfiltered_df.loc[shared_loops, shared_motifs]
    count_11      = count_11.loc[shared_loops]

    # --- compute difference ---
    diff_df = neural_df - unfiltered_df  # (n_loops, n_motifs)

    # --- fraction of motifs with positive difference (ignoring NaN) ---
    frac_positive = (diff_df > 0).sum(axis=1) / diff_df.notna().sum(axis=1)
    frac_positive.name = "frac_positive"

    # --- assemble output: frac_positive | count_11 | motif columns ---
    result_df = pd.concat([frac_positive, count_11, diff_df], axis=1)

    # --- STEP 1: Add loop presence binary columns ---
    loop_presence = load_loop_presence(window)
    result_df_with_presence, kept_loops = add_loop_presence_columns(result_df, loop_presence)
    
    # --- Calculate mean enrichment per motif (before dropping loops) ---
    # Get only motif columns (exclude tissues, frac_positive, count_11)
    known_cols = set(TISSUES) | {'frac_positive', 'count_11'}
    motif_columns = [col for col in result_df_with_presence.columns if col not in known_cols]
    
    if len(motif_columns) == 0:
        print_timestamp(f"Warning: No motif columns found for window {window}")
        return
    
    # Compute mean enrichment across loops for each motif
    mean_enrichment = result_df_with_presence[motif_columns].mean(axis=0)
    
    # --- STEP 2: Drop missing loops (NaN rows) ---
    result_df_clean = drop_missing_loops(result_df_with_presence)
    
    # --- Save enrichment table with loop presence columns ---
    out_path = os.path.join(output_dir, f"motif_enrichment_difference_hrs{window}.csv")
    result_df_clean.to_csv(out_path)

    # --- STEP 3: Create and save motif importance table ---
    motif_importance_df = create_motif_importance_table(window, motif_columns, mean_enrichment)
    
    importance_dir = os.path.join(output_dir,"motif_importance")
    os.makedirs(importance_dir, exist_ok=True)
    motif_importance_path = os.path.join(importance_dir, f"motif_importance_hrs{window}.csv")
    motif_importance_df.to_csv(motif_importance_path)


def main():
    for window in WINDOWS:
        compute_difference_for_window(window)
    print_timestamp("Computed enrichment differences and motif importance tables for all windows.")


if __name__ == "__main__":
    main()