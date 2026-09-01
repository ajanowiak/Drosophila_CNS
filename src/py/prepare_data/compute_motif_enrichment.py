# prapare_data/compute_motif_enrichment.py

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from core.log import configure_logging
from core.constants import WINDOWS, WINDOWS_PREV, FilteringMode
from prepare_data.io import load_window, load_metadata, save_enrichment_matrix, save_global_count_table

logger = logging.getLogger(__name__)

def compute_group_enrichment(
    loops_df: pd.DataFrame,
    motifs_df: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Compute motif enrichment for a single group of cells.

    Examples of cell groups used in this project:
        all cells, "neural_labels", "Epidermis", "Hindgut", ...

    Enrichment is defined as the difference between the mean chromVAR score
    among cells with both loop anchors open ("1-1" accessibility profile) and
    the mean score among all remaining cells.

    Args:
        loops_df:
            loops x cells accessibility profile matrix.

        motifs_df:
            motifs x cells chromVAR score matrix.
        
    Returns:
        A tuple containing

        - enrichment_df: DataFrame of shape (n_loops, n_motifs)
        - count_11: A vector with the number of "1-1" cells each loop has (n_loops,)
    """
    loop_ids = loops_df.index
    motif_ids = motifs_df.index

    n_loops = len(loop_ids)
    n_motifs = len(motif_ids)

    sum_11 = np.zeros((n_loops, n_motifs), dtype=np.float64)
    count_11 = np.zeros(n_loops, dtype=np.int64)

    sum_other = np.zeros((n_loops, n_motifs), dtype=np.float64)
    count_other = np.zeros(n_loops, dtype=np.int64)

    # Convert once to avoid repeated DataFrame indexing inside the loop.
    loops_mat = loops_df.to_numpy()
    motifs_mat = motifs_df.to_numpy()

    for i in range(n_loops):
        mask_11 = loops_mat[i] == 11
        mask_other = ~mask_11

        motifs_11 = motifs_mat[:, mask_11]
        motifs_other = motifs_mat[:, mask_other]

        if motifs_11.shape[1] > 0:
            sum_11[i] += motifs_11.sum(axis=1)
            count_11[i] += motifs_11.shape[1]

        if motifs_other.shape[1] > 0:
            sum_other[i] += motifs_other.sum(axis=1)
            count_other[i] += motifs_other.shape[1]

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_11 = np.where(
            count_11[:, None] > 0,
            sum_11 / count_11[:, None],
            np.nan,
        )

        mean_other = np.where(
            count_other[:, None] > 0,
            sum_other / count_other[:, None],
            np.nan,
        )

    enrichment_df = pd.DataFrame(
        mean_11 - mean_other,
        index=loop_ids,
        columns=motif_ids,
    )

    return enrichment_df, count_11


def compute_enrichment_for_window(
    window: str,
    filtering_mode: FilteringMode,
    metadata_df: pd.DataFrame | None,
    output_dir: Path,
) -> None:
    """
    Compute motif enrichment for a single developmental window.
    """

    logger.info("Loading input matrices...")
    grouped_data = load_window(
        window=window,
        filtering_mode=filtering_mode,
        metadata_df=metadata_df,
    )

    if not grouped_data:
        raise ValueError(
            f"No cell groups found for window {window} with filtering mode {filtering_mode}."
        )

    global_count_11: pd.DataFrame | None = None

    for group, (loops_df, motifs_df) in grouped_data.items():

        enrichment_df, count_11 = compute_group_enrichment(
            loops_df,
            motifs_df,
        )

        save_enrichment_matrix(
            enrichment_df=enrichment_df,
            group=group,
            window=window,
            output_dir=output_dir,
            filtering_mode=filtering_mode,
        )

        if global_count_11 is None:
            global_count_11 = pd.DataFrame(index=enrichment_df.index)

        global_count_11[group] = count_11

        logger.info("Saved group '%s'.", group)

    # grouped_data is non-empty (checked above), so the loop ran at least
    # once and global_count_11 was assigned a DataFrame.
    assert global_count_11 is not None
    save_global_count_table(
        global_count_df=global_count_11,
        window=window,
        output_dir=output_dir,
    )

    logger.info("Finished window %s.", window)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-loop motif enrichment for a single developmental time window."
    )

    parser.add_argument(
        "--window",
        required=True,
        # includes the preceding windows too (WINDOWS_PREV): stack_windows()'s
        # PREVIOUS/EXPANDED feature modes need enrichment matrices for those
        # as well as the three main windows.
        choices=WINDOWS + WINDOWS_PREV,
        help="Time window to process.",
    )

    parser.add_argument(
        "--filtering_mode",
        type=FilteringMode,
        choices=list(FilteringMode),
        required=True,
        help="Grouping strategy used to compute motif enrichment.",
    )

    parser.add_argument(
        "--metadata_path",
        default="data/atac_meta.rds",
        help="Path to the RDS file containing cell metadata. "
             "Only read when --filtering_mode requires cell annotations.",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where the enrichment matrices and count table are written.",
    )

    parser.add_argument(
        "--log_path",
        type=Path,
        required=True,
        help="Path to the log file for this run.",
    )

    args = parser.parse_args()

    configure_logging(args.log_path)

    metadata_df = (
        load_metadata(args.metadata_path)
        if args.filtering_mode != FilteringMode.UNFILTERED
        else None
    )

    compute_enrichment_for_window(
        window=args.window,
        filtering_mode=args.filtering_mode,
        metadata_df=metadata_df,
        output_dir=args.output_dir,
    )

    logger.info("Finished.")


if __name__ == "__main__":
    main()
