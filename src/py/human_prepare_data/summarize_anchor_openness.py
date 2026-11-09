# human_prepare_data/summarize_anchor_openness.py

"""
Count, for every anchor, the analysis cells it is open in: in all cells and within
each cell class.

The anchor x cell openness matrix (anchor_openness.py) is a dense 0/1 table of
~2.4 GB per library, too slow to reload for exploratory plots. This script reduces
it once to one row per anchor, so the anchor-openness notebook reads only small
files. Prototype step of the per-library PFC workflow (pfc_libraries.smk); nothing
downstream of Stage 0 uses it.

Inputs:
  - anchor_openness.tsv (anchor x analysis cell, 0/1; rows keyed by chr, start, end)
  - cell_metadata.tsv (barcode + the cell-class column)
Outputs:
  - anchor_openness_summary.tsv: chr, start, end, then open-cell counts in column
    "all" and one column per class
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from core.log import configure_logging

logger = logging.getLogger(__name__)

INDEX_COLUMNS = ["chr", "start", "end"]


def cell_classes(openness_path: Path, metadata_path: Path, group_column: str) -> pd.Series:
    """Class of every cell column of the openness table, in column order."""
    barcodes = pd.read_csv(openness_path, sep="\t", nrows=0).columns.drop(INDEX_COLUMNS)
    meta = pd.read_csv(metadata_path, sep="\t", index_col="barcode")
    classes = meta[group_column].reindex(barcodes)
    if classes.isna().any():
        raise ValueError(f"{int(classes.isna().sum())} openness cells have no {group_column} in {metadata_path}")
    return classes


def count_open_cells(openness_path: Path, classes: pd.Series, chunk_size: int) -> pd.DataFrame:
    """Open-cell counts per anchor, in all cells and per class, reading the table in row chunks."""
    dtypes = {barcode: np.uint8 for barcode in classes.index}
    class_names = sorted(classes.unique())
    class_masks = {name: (classes == name).to_numpy() for name in class_names}

    summaries = []
    for chunk in pd.read_csv(openness_path, sep="\t", index_col=INDEX_COLUMNS, dtype=dtypes, chunksize=chunk_size):
        open_matrix = chunk.to_numpy()
        counts = {"all": open_matrix.sum(axis=1)}
        counts.update({name: open_matrix[:, mask].sum(axis=1) for name, mask in class_masks.items()})
        summaries.append(pd.DataFrame(counts, index=chunk.index))
        logger.info(f"Summarized {sum(len(s) for s in summaries)} anchors")

    return pd.concat(summaries)


def main() -> None:
    parser = argparse.ArgumentParser(description="Count open cells per anchor, overall and per cell class.")
    parser.add_argument("--anchor_openness", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--group_column", required=True, help="Cell-class column of the metadata.")
    parser.add_argument("--chunk_size", type=int, default=5000, help="Anchors read per chunk.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log_path", type=Path, required=True)
    args = parser.parse_args()

    configure_logging(args.log_path)
    classes = cell_classes(args.anchor_openness, args.metadata, args.group_column)
    logger.info(f"Cells per class: {classes.value_counts().to_dict()}")

    summary = count_open_cells(args.anchor_openness, classes, args.chunk_size)
    logger.info(f"Anchors: {len(summary)}, never open: {int((summary['all'] == 0).sum())}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, sep="\t")
    logger.info(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
