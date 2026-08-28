# prepare_data/extract_loop_presence_vectors.py

"""
Extracts the binary loop-presence vector (ML training target) for a single
tissue and developmental time window from the long/short-range loop TSV.

Pipeline context: runs alongside compute_motif_enrichment.py and writes into
the same per-window, per-filtering-mode directory, since the enrichment
matrix (explanatory variables) and the presence vector (target variable) are
loaded together for training. Loop presence itself does not depend on
filtering mode; filtering mode only selects the output directory.

Inputs:
  - data/long_and_short_range_loops_D_mel.tsv (loop coordinates plus binary
    tissue/window presence columns, indexed by loop_id)

Outputs:
  - results/training_data/<filtering_mode>/hrs<window>/y_<tissue>.csv
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from core.log import configure_logging
from core.constants import WINDOWS, FilteringMode

logger = logging.getLogger(__name__)


def window_to_tsv_label(window: str) -> str:
    """
    Convert a "06-08"-style window into the TSV's column label style ("6-8h").
    """
    start, end = window.split("-")
    return f"{int(start)}-{int(end)}h"


def extract_presence_vector(
    loops_path: Path,
    window: str,
    tissue: str,
) -> pd.Series:
    """
    Extract the binary loop-presence vector for a single tissue and window.

    Args:
        loops_path: Path to the long/short-range loop TSV.
        window: Developmental time window (e.g. "06-08").
        tissue: Tissue name exactly as it appears in the TSV (e.g. "Glia").

    Returns:
        Series of 0/1 presence values indexed by loop_id.
    """
    column = f"Dmel_{window_to_tsv_label(window)}_{tissue}"

    header = pd.read_csv(loops_path, sep="\t", nrows=0).columns
    if column not in header:
        raise KeyError(
            f"Column '{column}' not found in {loops_path}. "
            f"Check that '{tissue}' is a valid tissue name."
        )

    loops_df = pd.read_csv(loops_path, sep="\t", index_col="loop_id", usecols=["loop_id", column])

    return loops_df[column]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract the binary loop-presence vector (ML target) for a single tissue and time window."
    )

    parser.add_argument(
        "--window",
        required=True,
        choices=WINDOWS,
        help="Time window to process.",
    )

    parser.add_argument(
        "--tissue",
        required=True,
        help="Tissue name exactly as it appears in the loop TSV (e.g. Glia).",
    )

    parser.add_argument(
        "--filtering_mode",
        type=FilteringMode,
        choices=list(FilteringMode),
        required=True,
        help="Only determines the output directory; presence vectors do not "
             "depend on filtering mode.",
    )

    parser.add_argument(
        "--loops_path",
        type=Path,
        default=Path("data/long_and_short_range_loops_D_mel.tsv"),
        help="Path to the long/short-range loop TSV.",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to write y_<tissue>.csv to.",
    )

    parser.add_argument(
        "--log_path",
        type=Path,
        required=True,
        help="Path to the log file for this run.",
    )

    args = parser.parse_args()

    configure_logging(args.log_path)

    presence = extract_presence_vector(
        loops_path=args.loops_path,
        window=args.window,
        tissue=args.tissue,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"y_{args.tissue}.csv"
    presence.to_csv(output_path)

    logger.info("Saved loop presence vector to %s", output_path)


if __name__ == "__main__":
    main()
