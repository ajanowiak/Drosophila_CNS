# human_prepare_data/compute_loop_sums.py

"""
Per-loop chromVAR sums of one library, the additive part of Stage 0 enrichment.

Enrichment is mean(score over "1-1" cells) - mean(score over the other cells).
Both means are a sum divided by a count, and sums and counts add up over cells,
so the enrichment of cells pooled from several libraries follows exactly from each
library's sums and counts (run_stage0_pooled.py). This script computes them for one
library, with the same activity codes and NaN rule as run_stage0_human.py; chromVAR
scores stay those of the library's own run (its own peaks and background). It also
counts each loop's cells with the left and with the right anchor open, which pool
the same way (not used by the pooled Stage 0).

Prototype step of the pooled-cell models (pfc_pools.smk).

Inputs:
  - anchor_openness.tsv and motifs_chromvar.tsv of one library
  - <negatives>/loop_universe.tsv
Outputs:
  - loop_sums.npz: loop_ids, motifs, n_cells, and per loop (rows in loop_ids order)
    sum_11 and sum_other (loop x motif), count_11, open_left, open_right
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from core.log import configure_logging
from human_prepare_data.anchor_openness import loop_activity
from human_prepare_data.run_stage0_human import align_cells
from prepare_data.compute_motif_enrichment import masked_sums

logger = logging.getLogger(__name__)

INDEX_COLUMNS = ["chr", "start", "end"]


def load_openness(path: Path) -> pd.DataFrame:
    """Anchor x cell 0/1 matrix, read as uint8 to keep memory at one byte per entry."""
    cells = pd.read_csv(path, sep="\t", nrows=0).columns.drop(INDEX_COLUMNS)
    return pd.read_csv(path, sep="\t", index_col=INDEX_COLUMNS, dtype={cell: np.uint8 for cell in cells})


def loop_motif_sums(activity: np.ndarray, motifs_df: pd.DataFrame,
                    chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Sum each motif's scores over each loop's "1-1" cells and over its other cells.

    Same products and NaN rule as compute_group_enrichment: a NaN score makes every
    sum it enters NaN.

    Args:
        activity: loop x cell activity codes (11 / 10 / 1 / 0).
        motifs_df: motif x cell chromVAR scores, cells in the order of activity's columns.
        chunk_size: loops per matrix product; bounds the memory of the float masks.

    Returns:
        sum_11 and sum_other, both loop x motif.
    """
    motif_scores = motifs_df.to_numpy(dtype=np.float64).T   # cells x motifs
    nan_scores = np.isnan(motif_scores)
    motif_nan = nan_scores.astype(np.float64) if nan_scores.any() else None
    motif_scores = np.where(nan_scores, 0.0, motif_scores)

    n_loops, n_motifs = activity.shape[0], motif_scores.shape[1]
    sum_11 = np.empty((n_loops, n_motifs), dtype=np.float64)
    sum_other = np.empty((n_loops, n_motifs), dtype=np.float64)
    for start in range(0, n_loops, chunk_size):
        rows = slice(start, start + chunk_size)
        mask_11 = (activity[rows] == 11).astype(np.float64)
        sum_11[rows] = masked_sums(mask_11, motif_scores, motif_nan)
        sum_other[rows] = masked_sums(1.0 - mask_11, motif_scores, motif_nan)
        logger.info(f"Summed {min(start + chunk_size, n_loops)} / {n_loops} loops")

    return sum_11, sum_other


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-loop chromVAR sums over '1-1' and other cells of one library.")
    parser.add_argument("--anchor_openness", type=Path, required=True)
    parser.add_argument("--motifs", type=Path, required=True)
    parser.add_argument("--loop_table", type=Path, required=True)
    parser.add_argument("--chunk_size", type=int, default=4096, help="Loops per matrix product.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log_path", type=Path, required=True)
    args = parser.parse_args()

    configure_logging(args.log_path)
    openness_df = load_openness(args.anchor_openness)
    motifs_df = pd.read_csv(args.motifs, sep="\t", index_col=0)
    loop_table = pd.read_csv(args.loop_table, sep="\t")

    openness_df, motifs_df = align_cells(openness_df, motifs_df)
    activity_df = loop_activity(loop_table, openness_df)
    activity = activity_df.to_numpy()

    sum_11, sum_other = loop_motif_sums(activity, motifs_df, args.chunk_size)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        loop_ids=activity_df.index.to_numpy(dtype=str),
        motifs=motifs_df.index.to_numpy(dtype=str),
        n_cells=np.int64(activity.shape[1]),
        sum_11=sum_11,
        sum_other=sum_other,
        count_11=(activity == 11).sum(axis=1).astype(np.int64),
        open_left=(activity >= 10).sum(axis=1).astype(np.int64),
        open_right=(activity % 10 == 1).sum(axis=1).astype(np.int64),
    )
    logger.info(f"Wrote {args.output} ({len(activity_df)} loops x {len(motifs_df)} motifs, {activity.shape[1]} cells)")


if __name__ == "__main__":
    main()
