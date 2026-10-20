# human_prepare_data/anchor_openness.py

"""
Compute the anchor x cell openness matrix once, for every distinct loop anchor.

An anchor is open in a cell if at least one QC-filtered fragment overlaps it.
Openness depends only on the anchor interval and the cell, not on which loop
the anchor belongs to, so it is computed per distinct anchor rather than per
loop, and once for both negative modes. The anchors are pooled from the three
Rahman loop BEDs: every loop of either loop universe pairs two of them (the
other_conditions negatives are real loops, and the n2 negatives re-pair a
condition's own anchors). Fragments come from the SNAP /FM session (no BEDPE
needed) and are overlapped with the anchors by bedtools intersect.

Stage 0 then combines each loop's two anchors into the two-anchor code used by
the Drosophila Stage 0 (loop_activity below): 11 = both anchors open, 10 = left
only, 1 = right only, 0 = neither.

Inputs:
  - a GSE244618 .snap file, cell_metadata.tsv (in_analysis selects the cells)
  - data/human_prototype/raw/{fetal,glia,neuron}_loop.bed.gz
Outputs:
  - data/human_prototype/interim/anchor_openness.tsv
    (anchor x analysis cell, 0/1; rows keyed by chr, start, end)
"""

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from pybedtools import BedTool

from core.log import configure_logging
from human_prepare_data.build_loop_universe import condition_anchors, read_loop_bed
from human_prepare_data.snap_io import load_barcodes, load_fragments

logger = logging.getLogger(__name__)


def pooled_anchors(loop_beds: list[Path]) -> pd.DataFrame:
    """Distinct anchors of all real loops in the given loop BEDs, sorted by (chr, start, end)."""
    loops = pd.concat([read_loop_bed(path) for path in loop_beds], ignore_index=True)
    anchors = condition_anchors(loops)
    logger.info(f"{len(loops)} real loops ({2 * len(loops)} anchor ends) -> {len(anchors)} distinct anchors")
    return anchors


def anchor_open_matrix(anchors: pd.DataFrame, fragments_df: pd.DataFrame, n_cells: int) -> np.ndarray:
    """Binary anchor x cell matrix: 1 if any fragment overlaps the anchor in that cell.

    The fragments' cell column must already hold output column positions (0 .. n_cells - 1).
    """
    anchor_bed = BedTool.from_dataframe(anchors[["chr", "start", "end"]].assign(anchor_row=np.arange(len(anchors))))
    fragment_bed = BedTool.from_dataframe(fragments_df[["chrom", "start", "end", "cell"]])

    # Interval join: one line per (fragment, overlapping anchor). Fragments are -a
    # so bedtools streams them and only holds the small anchor set in memory.
    joined = fragment_bed.intersect(anchor_bed, wa=True, wb=True)

    open_matrix = np.zeros((len(anchors), n_cells), dtype=np.int8)
    if Path(joined.fn).stat().st_size > 0:
        # Columns: fragment chrom start end cell | anchor chr start end anchor_row.
        # One overlap is enough, so collapse to distinct (anchor, cell) pairs.
        pairs = (pd.read_csv(joined.fn, sep="\t", header=None, usecols=[3, 7])
                 .set_axis(["cell", "anchor_row"], axis=1)
                 .drop_duplicates())
        open_matrix[pairs["anchor_row"].to_numpy(), pairs["cell"].to_numpy()] = 1

    return open_matrix


def compute_openness(snap_path: Path, anchors: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Return the anchor x cell openness matrix (0/1) for the analysis cells."""
    with h5py.File(snap_path, "r") as f:
        barcodes = load_barcodes(f)
        assert list(meta["barcode"]) == list(barcodes), "metadata not in snap cell order"

        cell_mask = meta["in_analysis"].to_numpy(dtype=bool)
        fragments_df = load_fragments(f, cell_mask=cell_mask)

    # Map each analysis cell's snap index to its column in the output matrix.
    selected_cell_indices = np.flatnonzero(cell_mask)
    column_of_cell = np.full(len(barcodes), -1, dtype=np.int64)
    column_of_cell[selected_cell_indices] = np.arange(len(selected_cell_indices))
    fragments_df["cell"] = column_of_cell[fragments_df["cell"].to_numpy()]
    logger.info(f"Anchors: {len(anchors)}, cells: {len(selected_cell_indices)}, fragments: {len(fragments_df)}")

    logger.info("Computing anchor openness ...")
    open_matrix = anchor_open_matrix(anchors, fragments_df, len(selected_cell_indices))
    n_open = int((open_matrix.sum(axis=1) > 0).sum())
    logger.info(f"Anchors open in >=1 cell: {n_open} / {len(anchors)}; "
                f"open (anchor, cell) fraction: {open_matrix.mean():.4f}")

    index = pd.MultiIndex.from_frame(anchors[["chr", "start", "end"]])
    return pd.DataFrame(open_matrix, index=index, columns=barcodes[selected_cell_indices])


def loop_activity(loop_table: pd.DataFrame, anchor_open: pd.DataFrame) -> pd.DataFrame:
    """Combine each loop's two anchors into the loop x cell activity code (11 / 10 / 1 / 0).

    Args:
        loop_table: loop_universe.tsv rows (chr1, x1, x2, y1, y2, loop_id, ...).
        anchor_open: anchor x cell 0/1 matrix indexed by (chr, start, end).

    Returns:
        loop x cell int8 DataFrame indexed by loop_id, with anchor_open's cell columns.
    """
    left_rows = anchor_open.index.get_indexer(pd.MultiIndex.from_frame(loop_table[["chr1", "x1", "x2"]]))
    right_rows = anchor_open.index.get_indexer(pd.MultiIndex.from_frame(loop_table[["chr1", "y1", "y2"]]))

    # Every loop must pair two pooled anchors; a miss means the anchor table and
    # the loop universe were built from different loop calls.
    missing = (left_rows < 0) | (right_rows < 0)
    if missing.any():
        raise ValueError(f"{int(missing.sum())} loops have an anchor missing from the anchor table "
                         f"(e.g. {loop_table.loc[missing, 'loop_id'].iloc[0]})")

    open_matrix = anchor_open.to_numpy(dtype=np.int8)
    activity_codes = (10 * open_matrix[left_rows] + open_matrix[right_rows]).astype(np.int8)

    code_counts = {int(code): int((activity_codes == code).sum()) for code in (11, 10, 1, 0)}
    logger.info(f"Activity code counts: {code_counts}")
    n_both_open = int(((activity_codes == 11).sum(axis=1) > 0).sum())
    logger.info(f"Loops with >=1 both-open cell: {n_both_open} / {len(loop_table)}")

    index = pd.Index(loop_table["loop_id"].to_numpy(), name="loop_id")
    return pd.DataFrame(activity_codes, index=index, columns=anchor_open.columns)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute the anchor x cell openness matrix from SNAP fragments.")
    parser.add_argument("--snap", type=Path, required=True)
    parser.add_argument("--loop_beds", type=Path, nargs="+", required=True,
                        help="The Rahman per-condition loop BEDs whose anchors are pooled.")
    parser.add_argument("--metadata", type=Path, default=Path("data/human_prototype/interim/cell_metadata.tsv"))
    parser.add_argument("--output", type=Path, default=Path("data/human_prototype/interim/anchor_openness.tsv"))
    parser.add_argument("--log_path", type=Path, default=Path("logs/human_prototype/anchor_openness.log"))
    args = parser.parse_args()

    configure_logging(args.log_path)
    anchors = pooled_anchors(args.loop_beds)
    meta = pd.read_csv(args.metadata, sep="\t")

    openness = compute_openness(args.snap, anchors, meta)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    openness.to_csv(args.output, sep="\t")
    logger.info(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
