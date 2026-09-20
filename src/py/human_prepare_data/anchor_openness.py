# human_prepare_data/anchor_openness.py

"""
Compute the loop x cell anchor-activity matrix for the selected nuclei.

An anchor is open in a cell if at least one QC-filtered fragment overlaps it;
each loop's activity in a cell is the two-anchor code used by the Drosophila
Stage 0: 11 = both anchors open, 10 = left only, 1 = right only, 0 = neither.
Fragments come from the SNAP /FM session (no BEDPE needed).

Inputs:
  - a GSE244618 .snap file, loop_universe.tsv, cell_metadata.tsv
Outputs:
  - data/human_prototype/interim/loops_activity.tsv  (loop x cell, in {0,1,10,11})
"""

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyranges as pr

from core.log import configure_logging
from human_prepare_data.snap_io import load_barcodes, load_fragments

logger = logging.getLogger(__name__)


def anchor_open_matrix(loops: pd.DataFrame, fragments_pr: pr.PyRanges, side: str,
                       loop_pos: dict, cell_pos: dict, shape: tuple[int, int]) -> np.ndarray:
    """Boolean loop x cell matrix: 1 if any fragment overlaps this loop's anchor in that cell."""
    start_col, end_col = ("x1", "x2") if side == "left" else ("y1", "y2")

    anchor_pr = pr.PyRanges(loops[["chr1", start_col, end_col, "loop_id"]].rename(
        columns={"chr1": "Chromosome", start_col: "Start", end_col: "End"}))

    # Interval join: for each anchor, every fragment whose span overlaps it.
    joined = anchor_pr.join(fragments_pr)

    open_matrix = np.zeros(shape, dtype=np.int8)
    if len(joined):
        # One overlap is enough, so collapse to distinct (loop, cell) pairs.
        pairs = joined.df[["loop_id", "Cell"]].drop_duplicates()
        rows = pairs["loop_id"].map(loop_pos).to_numpy()
        cols = pairs["Cell"].map(cell_pos).to_numpy()
        open_matrix[rows, cols] = 1

    return open_matrix


def compute_activity(snap_path: Path, loops: pd.DataFrame, meta: pd.DataFrame,
                     cell_label: str, label_column: str = "label") -> pd.DataFrame:
    """Return the loop x cell activity matrix (codes 0/1/10/11) for the chosen cells."""
    loop_pos = {loop_id: i for i, loop_id in enumerate(loops["loop_id"])}

    with h5py.File(snap_path, "r") as f:
        barcodes = load_barcodes(f)
        assert list(meta["barcode"]) == list(barcodes), "metadata not in snap cell order"

        cell_selection = meta["pass_qc"] if cell_label == "all" else \
            (meta["pass_qc"] & (meta[label_column] == cell_label))
        cell_mask = cell_selection.to_numpy()
        fragments_df = load_fragments(f, cell_mask=cell_mask)

    # Map each selected cell's snap index to its column in the output matrix.
    selected_cell_indices = np.flatnonzero(cell_mask)
    cell_pos = {int(cell_index): col for col, cell_index in enumerate(selected_cell_indices)}
    shape = (len(loops), len(selected_cell_indices))
    logger.info(f"Loops: {shape[0]}, cells: {shape[1]}, fragments: {len(fragments_df)}")

    fragments_pr = pr.PyRanges(fragments_df.rename(
        columns={"chrom": "Chromosome", "start": "Start", "end": "End", "cell": "Cell"}))

    logger.info("Computing anchor openness ...")
    open_left = anchor_open_matrix(loops, fragments_pr, "left", loop_pos, cell_pos, shape)
    open_right = anchor_open_matrix(loops, fragments_pr, "right", loop_pos, cell_pos, shape)

    # Combine the two anchors into one code: 11 / 10 / 1 / 0.
    activity_codes = (10 * open_left + open_right).astype(np.int8)

    code_counts = {int(code): int((activity_codes == code).sum()) for code in (11, 10, 1, 0)}
    logger.info(f"Activity code counts: {code_counts}")
    n_both_open = int(((activity_codes == 11).sum(axis=1) > 0).sum())
    logger.info(f"Loops with >=1 both-open cell: {n_both_open} / {shape[0]}")

    return pd.DataFrame(activity_codes, index=loops["loop_id"], columns=barcodes[selected_cell_indices])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute the loop x cell anchor-activity matrix from SNAP fragments.")
    parser.add_argument("--snap", type=Path, required=True)
    parser.add_argument("--loops", type=Path, default=Path("data/human_prototype/interim/loop_universe.tsv"))
    parser.add_argument("--metadata", type=Path, default=Path("data/human_prototype/interim/cell_metadata.tsv"))
    parser.add_argument("--cell_label", default="neuron")
    parser.add_argument("--label_column", default="label",
                        help="cell_metadata column to match --cell_label against "
                             "(label / cellclass / subclass / celltype).")
    parser.add_argument("--output", type=Path, default=Path("data/human_prototype/interim/loops_activity.tsv"))
    parser.add_argument("--log_path", type=Path, default=Path("logs/human_prototype/anchor_openness.log"))
    args = parser.parse_args()

    configure_logging(args.log_path)
    loops = pd.read_csv(args.loops, sep="\t")
    meta = pd.read_csv(args.metadata, sep="\t")

    activity = compute_activity(args.snap, loops, meta, args.cell_label, args.label_column)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    activity.to_csv(args.output, sep="\t")
    logger.info(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
