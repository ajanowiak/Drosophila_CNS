# generate_motif_enrichment_with_filtering.py

import os
import argparse
import numpy as np
import pandas as pd
import pyreadr
import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils import load_window_split_by_tissue, print_timestamp


WINDOWS = ['06-08', '10-12', '14-16']
ACTIVITY_PROFILES = ["1-1", "1-0", "0-1", "0-0"]

NEURAL_LABELS_RAW = [
    "Brain", "Neural", "Ventral_nerve_cord",
    "Ventral_nerve_cord_prim", "Glia", "PNS_&_sense"
]
NEURAL_LABELS = list(map(lambda s: s.replace("prim", "prim.").replace("_", " "), NEURAL_LABELS_RAW))

# def compute_enrichment_for_window(window: str, anot_df: pd.DataFrame, filter_neural: bool = False) -> None:
#     """
#     For a single time window:
#       - load data split by tissue
#       - for every tissue compute per-loop motif enrichment (mean_11 - mean_other) fully vectorized

#     No intermediate distribution storage: we accumulate weighted sums directly.
#     """
#     print_timestamp(f"Window {window}: loading tissue-split data...")
#     tissue_dict = load_window_split_by_tissue(window=window, metadata_df=anot_df)

#     if filter_neural:
#         valid_labels = [l for l in NEURAL_LABELS if l in tissue_dict]
#         if not valid_labels:
#             print_timestamp(f"Window {window}: no valid neural labels found, skipping.")
#             return
#     else:
#         valid_labels = list(tissue_dict.keys())

#     # Retrieve index labels from the first valid tissue
#     first_loops_df, first_motifs_df = tissue_dict[valid_labels[0]]
#     loop_ids = list(first_loops_df.index)
#     motif_ids = list(first_motifs_df.index)

#     # For each loop, accumulate sum and count across all neural tissues
#     # Shape: (n_loops, n_motifs)
#     n_loops = len(loop_ids)
#     n_motifs = len(motif_ids)

#     sum_11    = np.zeros((n_loops, n_motifs), dtype=np.float64)
#     count_11  = np.zeros((n_loops,),          dtype=np.int64)
#     sum_other    = np.zeros((n_loops, n_motifs), dtype=np.float64)
#     count_other  = np.zeros((n_loops,),          dtype=np.int64)

#     for label in valid_labels:
#         loops_df, motifs_df = tissue_dict[label]

#         # numpy matrices: shape (n_loops, n_cells) and (n_motifs, n_cells)
#         loops_mat  = loops_df.to_numpy()   # (n_loops,  n_cells)
#         motifs_mat = motifs_df.to_numpy() # (n_motifs, n_cells)

#         for i in range(n_loops):
#             mask_11    = loops_mat[i] == 11
#             mask_other = ~mask_11  # 1-0, 0-1, 0-0 combined

#             motifs_11    = motifs_mat[:, mask_11]    # (n_motifs, n_cells_11)
#             motifs_other = motifs_mat[:, mask_other] # (n_motifs, n_cells_other)

#             if motifs_11.shape[1] > 0:
#                 sum_11[i]   += motifs_11.sum(axis=1)
#                 count_11[i] += motifs_11.shape[1]

#             if motifs_other.shape[1] > 0:
#                 sum_other[i]   += motifs_other.sum(axis=1)
#                 count_other[i] += motifs_other.shape[1]

#         # i to jest indeks pętli !
#         # tutaj liczenie mean i różnicy, zapisywanie .csv dla pojedynczych tkanek

#     # Vectorized mean difference: (n_loops, n_motifs)
#     with np.errstate(invalid='ignore', divide='ignore'):
#         mean_11    = np.where(count_11[:, None]    > 0, sum_11    / count_11[:, None],    np.nan)
#         mean_other = np.where(count_other[:, None] > 0, sum_other / count_other[:, None], np.nan)

#     enrichment_matrix = mean_11 - mean_other  # (n_loops, n_motifs)

#     # Save one CSV per loop (matching original output format)
#     if filter_neural:
#         output_dir = f"results/training_data/neural_labels/hrs{window}"
#     else:
#         output_dir = f"results/training_data/unfiltered/hrs{window}"
    
#     os.makedirs(output_dir, exist_ok=True)

#     # Save the enrichment matrix
#     full_table = pd.DataFrame(enrichment_matrix, index=loop_ids, columns=motif_ids, dtype=float)
#     full_path = os.path.join(output_dir, f"motif_enrichment_hrs{window}.csv")
#     full_table.to_csv(full_path)
#     print_timestamp(f"Window {window}: saved enrichment matrix to {full_path}")

#     # Save the 1-1 cell counts matrix (same shape as enrichment table)
#     # count_11[i] is the number of 1-1 cells for loop i, identical across all motifs
#     count_11_matrix = np.broadcast_to(count_11[:, None], (n_loops, n_motifs))
#     count_table = pd.DataFrame(count_11_matrix, index=loop_ids, columns=motif_ids, dtype=np.int64)
#     count_path = os.path.join(output_dir, f"count11_hrs{window}.csv")
#     count_table.to_csv(count_path)
#     print_timestamp(f"Window {window}: saved 1-1 cell counts matrix to {count_path}")


def compute_enrichment_for_window(window: str, anot_df: pd.DataFrame, filter_neural: bool = False) -> None:
    """
    For a single time window:
      - load data split by tissue
      - compute per-loop motif enrichment (mean_11 - mean_other) separately for each tissue
      - save:
          (1) one enrichment matrix per tissue
          (2) one count_11 vector per tissue
          (3) one global count_11 table (loops x tissues)

    No intermediate distribution storage: we accumulate weighted sums directly.
    """
    print_timestamp(f"Window {window}: loading tissue-split data...")
    tissue_dict = load_window_split_by_tissue(window=window, metadata_df=anot_df)

    if filter_neural:
        valid_labels = [l for l in NEURAL_LABELS if l in tissue_dict]
        if not valid_labels:
            print_timestamp(f"Window {window}: no valid neural labels found, skipping.")
            return
    else:
        valid_labels = list(tissue_dict.keys())

    # Output directory
    if filter_neural:
        output_dir = f"results/training_data/neural_labels/hrs{window}"
    else:
        output_dir = f"results/training_data/refined_annotations/hrs{window}"

    os.makedirs(output_dir, exist_ok=True)

    # Initialize global count table (loops x tissues)
    global_count_11 = None

    for label in valid_labels:
        # print_timestamp(f"Window {window}: processing tissue '{label}'")

        loops_df, motifs_df = tissue_dict[label]

        loop_ids = list(loops_df.index)
        motif_ids = list(motifs_df.index)

        n_loops = len(loop_ids)
        n_motifs = len(motif_ids)

        # Initialize per-tissue accumulators
        sum_11    = np.zeros((n_loops, n_motifs), dtype=np.float64)
        count_11  = np.zeros((n_loops,),          dtype=np.int64)
        sum_other = np.zeros((n_loops, n_motifs), dtype=np.float64)
        count_other = np.zeros((n_loops,),        dtype=np.int64)

        # Convert to numpy once
        loops_mat  = loops_df.to_numpy()   # (n_loops, n_cells)
        motifs_mat = motifs_df.to_numpy() # (n_motifs, n_cells)

        for i in range(n_loops):
            mask_11    = loops_mat[i] == 11
            mask_other = ~mask_11

            motifs_11    = motifs_mat[:, mask_11]
            motifs_other = motifs_mat[:, mask_other]

            if motifs_11.shape[1] > 0:
                sum_11[i]   += motifs_11.sum(axis=1)
                count_11[i] += motifs_11.shape[1]

            if motifs_other.shape[1] > 0:
                sum_other[i]   += motifs_other.sum(axis=1)
                count_other[i] += motifs_other.shape[1]

        # Compute means
        with np.errstate(invalid='ignore', divide='ignore'):
            mean_11    = np.where(count_11[:, None]    > 0, sum_11    / count_11[:, None],    np.nan)
            mean_other = np.where(count_other[:, None] > 0, sum_other / count_other[:, None], np.nan)

        enrichment_matrix = mean_11 - mean_other

        # --- Save enrichment matrix ---
        enrichment_df = pd.DataFrame(enrichment_matrix, index=loop_ids, columns=motif_ids)
        enrichment_path = os.path.join(output_dir, f"{label}_motif_enrichment_hrs{window}.csv")
        enrichment_df.to_csv(enrichment_path)

        # --- Accumulate into global table ---
        if global_count_11 is None:
            global_count_11 = pd.DataFrame(index=loop_ids)

        global_count_11[label] = count_11

        print_timestamp(f"Window {window}: saved tissue '{label}' outputs")

    # --- Save global counts table (loops x tissues) ---
    global_count_path = os.path.join(output_dir, f"count11_all_tissues_hrs{window}.csv")
    global_count_11.to_csv(global_count_path)

    print_timestamp(f"Window {window}: saved global count table to {global_count_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filter_neural",
        type=lambda x: x.lower() == "true",
        required=True,
        help="Whether to filter based on neural labels (True/False)." \
        " Choosing 'False' will result in splitting all data by refined_annotation and producing a separate enrichment matrix for every tissue annotation."
    )
    
    args = parser.parse_args()

    print_timestamp("Reading metadata (tissue annotations)...")
    atac_meta = pyreadr.read_r('data/atac_meta.rds')
    anot_df = list(atac_meta.values())[0]

    # Process windows in parallel (one process per window)
    with ProcessPoolExecutor(max_workers=len(WINDOWS)) as executor:
        futures = {
            executor.submit(compute_enrichment_for_window, w, anot_df, args.filter_neural): w
            for w in WINDOWS
        }
        for fut in as_completed(futures):
            w = futures[fut]
            try:
                fut.result()
            except Exception as e:
                print_timestamp(f"\t Window {w} failed: {e}")

    print_timestamp("All tables saved.")


if __name__ == '__main__':
    main()