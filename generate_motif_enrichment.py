# generate_motif_enrichment.py

import os
import numpy as np
import pandas as pd
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils import load_window, print_timestamp

WINDOWS = ["06-08", "10-12", "14-16"]
WINDOWS_PREV = ["04-06", "08-10", "12-14"]

def compute_enrichment_for_window(window: str) -> None:
    print_timestamp(f"Window {window}: loading data...")
    loops_df, motifs_df = load_window(window)

    loop_ids = list(loops_df.index)
    motif_ids = list(motifs_df.index)
    n_loops = len(loop_ids)

    loops_mat = loops_df.to_numpy()   # (n_loops,  n_cells)
    motifs_mat = motifs_df.to_numpy() # (n_motifs, n_cells)

    n_motifs = motifs_mat.shape[0]

    sum_11      = np.zeros((n_loops, n_motifs), dtype=np.float64)
    count_11    = np.zeros((n_loops,),          dtype=np.int64)
    sum_other   = np.zeros((n_loops, n_motifs), dtype=np.float64)
    count_other = np.zeros((n_loops,),          dtype=np.int64)

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

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_11    = np.where(count_11[:, None]    > 0, sum_11    / count_11[:, None],    np.nan)
        mean_other = np.where(count_other[:, None] > 0, sum_other / count_other[:, None], np.nan)

    enrichment_matrix = mean_11 - mean_other

    output_dir = f"results/training_data/unfiltered/hrs{window}"
    os.makedirs(output_dir, exist_ok=True)

    enrichment_df = pd.DataFrame(enrichment_matrix, index=loop_ids, columns=motif_ids)
    enrichment_df.to_csv(os.path.join(output_dir, f"motif_enrichment_hrs{window}.csv"))

    count_11_matrix = np.broadcast_to(count_11[:, None], (n_loops, n_motifs))
    count_df = pd.DataFrame(count_11_matrix, index=loop_ids, columns=motif_ids, dtype=np.int64)
    count_df.to_csv(os.path.join(output_dir, f"count11_hrs{window}.csv"))

    print_timestamp(f"Window {window}: done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_current_windows",
        type=lambda x: x.lower() == "true",
        default=True,
        help='Choose True for ["06-08", "10-12", "14-16"] (current windows) and False for ["04-06", "08-10", "12-14"] (previous windows)'
    )
    
    args = parser.parse_args()

    if args.use_current_windows:
        window_list = WINDOWS
    else:
        window_list = WINDOWS_PREV
    
    with ProcessPoolExecutor(max_workers=len(window_list)) as executor:
        futures = {executor.submit(compute_enrichment_for_window, w): w for w in window_list}
        for fut in as_completed(futures):
            w = futures[fut]
            try:
                fut.result()
            except Exception as e:
                print_timestamp(f"Window {w} failed: {e}")

    print_timestamp("All tables saved.")


if __name__ == "__main__":
    main()