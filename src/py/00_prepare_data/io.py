# 00_prepare_data/io.py

import logging
from pathlib import Path
import pandas as pd
import pyreadr

from core.constants import FilteringMode, NEURAL_LABELS

logger = logging.getLogger(__name__)

def load_window(
    window: str,
    filtering_mode: FilteringMode = FilteringMode.UNFILTERED,
    metadata_df: pd.DataFrame | None = None,
    data_dir: Path = Path("data/new_time"),
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Load chromatin loop accessibility profile and motif chromVAR score matrices
    for a single developmental time window.

    The returned object is always a dictionary mapping a label to a
    ``(loops_df, motifs_df)`` tuple. This provides a uniform interface for
    downstream enrichment calculations regardless of whether the analysis is
    performed on all cells or on annotation-specific subsets.

    Filtering modes
    ---------------
    UNFILTERED
        Returns a single entry named ``"unfiltered"`` containing all cells.

    NEURAL_LABELS
        Returns a single entry named ``neural_labels`` containing all cells
        with the neural label.

    REFINED_ANNOTATIONS
        Returns one entry for every refined cell annotation present in
        ``metadata_df``.

    Args:
        window:
            Developmental time window (e.g. ``"06-08"``).

        filtering_mode:
            Strategy used to partition cells before enrichment analysis.

        metadata_df:
            Cell metadata containing a ``refined_annotation`` column.
            Required for annotation-based filtering modes.

        data_dir:
            Directory containing the loop and motif matrices.

    Returns:
        Dictionary mapping group names to ``(loops_df, motifs_df)`` tuples.
    """

    def load_df(path: Path, label: str = 'loops') -> pd.DataFrame:

        logger.info(f"Loading {label} data for window {window}")
        df = pd.read_csv(loops_path, sep="\t", index_col=0)

        logger.info(f"Converting the {label} matrix to numeric values and removing invalid cells")
        df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1)

        return df

    loops_path = data_dir / f"hrs{window}_NNv1_time_matrix_loops.tsv"
    motifs_path = data_dir / f"hrs{window}_NNv1_time_matrix_motifs.tsv"

    loops_df = load_df(loops_path, label="loops")
    motifs_df = load_df(motifs_path, label="motifs")

    common_cells = loops_df.columns.intersection(motifs_df.columns)

    loops_df, motifs_df = loops_df[common_cells], motifs_df[common_cells]

    assert loops_df.columns.equals(motifs_df.columns), (
        "Loop and motif matrices contain different cell columns."
    )

    # expected filtering mode is UNFILTERED
    if filtering_mode == FilteringMode.UNFILTERED:
        return {
            "unfiltered": (
                loops_df,
                motifs_df,
            )
        }

    if metadata_df is None:
        raise ValueError(
            "metadata_df must be provided when using annotation-based filtering."
        )

    # the other filtering modes were only necessary for scripts that are now obsolete
    if filtering_mode == FilteringMode.NEURAL_LABELS:
        neural_cells = metadata_df.loc[
            metadata_df["refined_annotation"].isin(NEURAL_LABELS)
        ].index.intersection(common_cells)

        return {
            "neural_labels": (
                loops_df[neural_cells],
                motifs_df[neural_cells],
            )
        }

    if filtering_mode == FilteringMode.REFINED_ANNOTATIONS:
        grouped = {}

        for label, submeta in metadata_df.groupby("refined_annotation"):
            cells = submeta.index.intersection(common_cells)

            if len(cells) == 0:
                continue

            grouped[label] = (
                loops_df[cells],
                motifs_df[cells],
            )

        return grouped

    raise ValueError(f"Unknown filtering mode: {filtering_mode}")


def load_metadata(metadata_path: str = "data/atac_meta.rds") -> pd.DataFrame:
    """
    Load the scATAC-seq cell metadata.

    Args:
        metadata_path: Path to the RDS file containing cell metadata.

    Returns:
        Cell metadata indexed by cell barcode.
    """
    logger.info("Loading metadata...")

    metadata = pyreadr.read_r(metadata_path)

    return next(iter(metadata.values()))


def save_enrichment_matrix(
    enrichment_df: pd.DataFrame,
    group: str,
    window: str,
    output_dir: Path,
) -> None:
    """
    Save a motif enrichment matrix for a single group.

    Args:
        enrichment_df: Enrichment matrix (loops × motifs).
        group: Group name (tissue name, "all", etc.).
        window: Time window (e.g. "06-08").
        output_dir: Directory where enrichment matrices should be written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # sort loops according to numerical order instead of lexicographic order (eg. L2 before L10)
    enrichment_df = enrichment_df.sort_index(
        key=lambda idx: idx.astype(str).str.extract(r"(\d+)").astype(int)[0]
    )

    output_path = output_dir / f"{group}_motif_enrichment_hrs{window}.csv"
    enrichment_df.to_csv(output_path)


def save_global_count_table(
    global_count_df: pd.DataFrame,
    window: str,
    output_dir: Path,
) -> None:
    """
    Save the per-group counts of 1-1 cells for every loop.

    Args:
        global_count_df: Loop × group table of 1-1 cell counts.
        window: Time window (e.g. "06-08").
        output_dir: Directory where the count table should be written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"count11_all_tissues_hrs{window}.csv"
    global_count_df.to_csv(output_path)

