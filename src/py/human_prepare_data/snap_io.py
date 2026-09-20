# human_prepare_data/snap_io.py

"""
Readers for the SnapATAC v1 .snap HDF5 file (GSE244618 processed data).

Exposes the cell barcodes, the cell x peak matrix and peak coordinates (/PM,
for chromVAR), the cell x gene activity matrix (/GM, for annotation), and the
per-cell QC-filtered fragments (/FM, for anchor openness).
"""

import logging

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)


def _decode(arr) -> np.ndarray:
    """Turn an HDF5 array of bytes strings into a numpy array of str."""
    return np.array([x.decode() if isinstance(x, bytes) else x for x in arr])


def _load_coo(f: h5py.File, group: str, n_rows: int, n_cols: int) -> csr_matrix:
    """Load a SNAP sparse session (1-based idx/idy/count) as a cell x feature CSR."""
    idx = f[f"/{group}/idx"][:].astype(np.int64) - 1
    idy = f[f"/{group}/idy"][:].astype(np.int64) - 1
    val = f[f"/{group}/count"][:].astype(np.float32)
    return csr_matrix((val, (idx, idy)), shape=(n_rows, n_cols))


def load_barcodes(f: h5py.File) -> np.ndarray:
    """Cell barcodes, in SNAP cell order."""
    return _decode(f["/BD/name"][:])


def load_peaks(f: h5py.File) -> pd.DataFrame:
    """Peak coordinates as a chrom/start/end DataFrame."""
    return pd.DataFrame({
        "chrom": _decode(f["/PM/peakChrom"][:]),
        "start": f["/PM/peakStart"][:].astype(np.int64),
        "end": f["/PM/peakEnd"][:].astype(np.int64),
    })


def load_cell_peak(f: h5py.File) -> csr_matrix:
    """Cell x peak accessibility matrix."""
    return _load_coo(f, "PM", f["/BD/name"].shape[0], f["/PM/peakChrom"].shape[0])


def load_cell_gene(f: h5py.File) -> tuple[csr_matrix, np.ndarray]:
    """Cell x gene activity matrix and the gene names."""
    names = _decode(f["/GM/name"][:])
    return _load_coo(f, "GM", f["/BD/name"].shape[0], len(names)), names


def load_fragment_counts(f: h5py.File) -> np.ndarray:
    """Per-cell unique-fragment count (/FM/barcodeLen == /BD/UQ), a QC depth."""
    return f["/FM/barcodeLen"][:].astype(np.int64)


def load_fragments(f: h5py.File, cell_mask: np.ndarray | None = None) -> pd.DataFrame:
    """QC-filtered fragments with a 0-based cell index (chrom, start, end, cell).

    Fragments are stored as one long concatenated list, block by block per
    barcode; the per-cell index is reconstructed from the block lengths. If
    cell_mask (bool over cells) is given, only those cells' fragments are kept.
    """
    # /FM has no explicit cell column: it stores fragments grouped by barcode,
    # with barcodeLen[i] = how many fragments belong to cell i, and
    # barcodePos[i] = 1-based offset where cell i's block starts.
    frags_per_cell = f["/FM/barcodeLen"][:].astype(np.int64)
    barcode_pos = f["/FM/barcodePos"][:].astype(np.int64) - 1
    n_fragments = f["/FM/fragStart"].shape[0]

    # Sanity-check the layout before we trust it: the block lengths must sum to
    # the total, and the stated offsets must be their running cumulative sum
    # (i.e. the blocks really are contiguous and in barcode order).
    assert int(frags_per_cell.sum()) == n_fragments, "barcodeLen sum != number of fragments"
    assert np.array_equal(barcode_pos, np.concatenate([[0], np.cumsum(frags_per_cell)[:-1]])), \
        "fragment blocks are not in barcode order"

    start = f["/FM/fragStart"][:].astype(np.int64)
    fragments = pd.DataFrame({
        "chrom": _decode(f["/FM/fragChrom"][:]),
        "start": start,
        "end": start + f["/FM/fragLen"][:].astype(np.int64),
        # Expand the block lengths back into a per-fragment cell index.
        "cell": np.repeat(np.arange(len(frags_per_cell)), frags_per_cell),
    })

    if cell_mask is not None:
        fragments = fragments[fragments["cell"].isin(np.flatnonzero(cell_mask))].reset_index(drop=True)

    scope = "" if cell_mask is None else f" for {int(cell_mask.sum())} cells"
    logger.info(f"Loaded {len(fragments)} fragments{scope}")
    return fragments
