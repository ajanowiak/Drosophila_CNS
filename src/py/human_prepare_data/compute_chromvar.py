# human_prepare_data/compute_chromvar.py

"""
Compute a cell x motif chromVAR deviation matrix for the selected nuclei.

Runs the standard pychromvar workflow on hg38 with JASPAR CORE vertebrate
motifs: GC bias -> background peaks -> motif matching -> deviations.

Inputs:  a GSE244618 .snap file, hg38 fasta, cell_metadata.tsv
Outputs: motifs_chromvar.tsv (motif x cell)
"""

import argparse
import logging
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pychromvar as pc
from pyjaspar import jaspardb
from scipy.sparse import csr_matrix

from core.log import configure_logging
from human_prepare_data.constants import MAIN_CHROMS
from human_prepare_data.snap_io import load_barcodes, load_cell_peak, load_peaks

logger = logging.getLogger(__name__)


def build_adata(snap_path: Path, keep_barcodes: set[str], min_cells_per_peak: int,
                chroms: list[str] | None) -> ad.AnnData:
    """AnnData of the selected cells x accessible main-chromosome peaks.

    pychromvar operates on an AnnData (cells in obs, peaks in var, counts in X),
    so this packs the SNAP cell x peak matrix into that container after filtering
    to the chosen cells and to peaks accessible in enough of them.
    """
    with h5py.File(snap_path, "r") as f:
        barcodes_array = load_barcodes(f)        # 1-D array of barcode strings
        peaks_df = load_peaks(f)                 # DataFrame: chrom / start / end
        cell_peak_matrix = load_cell_peak(f)     # sparse CSR: cells x peaks

    cell_mask = np.isin(barcodes_array, list(keep_barcodes))
    cell_peak_matrix = cell_peak_matrix[cell_mask]
    kept_barcodes = barcodes_array[cell_mask]
    logger.info(f"Selected {cell_mask.sum()} cells")

    # Keep peaks that are on the main chromosomes AND accessible in enough cells;
    # the accessibility filter is also what bounds chromVAR's memory.
    allowed_chroms = set(chroms) if chroms else set(MAIN_CHROMS)
    on_main_chrom = peaks_df["chrom"].isin(allowed_chroms).to_numpy()
    cells_per_peak = np.asarray((cell_peak_matrix > 0).sum(axis=0)).ravel()
    accessible_enough = cells_per_peak >= min_cells_per_peak
    peak_mask = on_main_chrom & accessible_enough

    cell_peak_matrix = cell_peak_matrix[:, peak_mask]
    peaks_df = peaks_df.loc[peak_mask].reset_index(drop=True)
    logger.info(f"Kept {peak_mask.sum()} peaks (accessible in >= {min_cells_per_peak} cells)")

    # AnnData's per-peak annotation frame (its .var); index each peak as
    # "chrom-start-end" so add_peak_seq can split it back on the "-" delimiter.
    peak_annotations = peaks_df.rename(columns={"chrom": "chr"})
    peak_annotations.index = [f"{c}-{s}-{e}"
                              for c, s, e in zip(peaks_df["chrom"], peaks_df["start"], peaks_df["end"])]

    return ad.AnnData(X=csr_matrix(cell_peak_matrix),
                      obs=pd.DataFrame(index=kept_barcodes),
                      var=peak_annotations)


def run_chromvar(adata: ad.AnnData, genome_file: str, jaspar_release: str,
                 bg_iterations: int) -> pd.DataFrame:
    """Run the pychromvar workflow and return a motif x cell deviation matrix."""
    logger.info("add_peak_seq / add_gc_bias ...")
    pc.add_peak_seq(adata, genome_file=genome_file, delimiter="-")
    pc.add_gc_bias(adata)

    logger.info(f"get_bg_peaks (niterations={bg_iterations}) ...")
    pc.get_bg_peaks(adata, niterations=bg_iterations, n_jobs=1)

    motifs = jaspardb(release=jaspar_release).fetch_motifs(collection="CORE", tax_group=["vertebrates"])
    logger.info(f"matching {len(motifs)} JASPAR {jaspar_release} motifs ...")
    pc.match_motif(adata, motifs=motifs, genome_file=genome_file)

    # n_jobs=1 on purpose: parallel workers each pickle a dense copy of the
    # cell x peak matrix and exhaust memory; BLAS threads the matmuls instead.
    logger.info("compute_deviations (n_jobs=1; BLAS-threaded) ...")
    deviations = pc.compute_deviations(adata, n_jobs=1)

    deviation_df = pd.DataFrame(np.asarray(deviations.X).T,
                                index=deviations.var_names, columns=deviations.obs_names)
    logger.info(f"chromVAR deviations: {deviation_df.shape[0]} motifs x {deviation_df.shape[1]} cells")
    return deviation_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute chromVAR cell x motif deviations from SNAP peaks.")
    parser.add_argument("--snap", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=Path("data/human_prototype/interim/cell_metadata.tsv"))
    parser.add_argument("--cell_label", default="neuron",
                        help="value to keep in --label_column, or 'all' for every QC nucleus.")
    parser.add_argument("--label_column", default="label",
                        help="cell_metadata column to match --cell_label against "
                             "(label / cellclass / subclass / celltype).")
    parser.add_argument("--genome", type=Path, default=Path("data/human_prototype/ref/hg38.fa"))
    parser.add_argument("--jaspar_release", default="JASPAR2026")
    parser.add_argument("--min_cells_per_peak", type=int, default=25,
                        help="Accessibility filter; higher => fewer peaks => less memory.")
    parser.add_argument("--bg_iterations", type=int, default=25, help="chromVAR background peak sets.")
    parser.add_argument("--chroms", nargs="*", default=None, help="Optional chrom subset (smoke test).")
    parser.add_argument("--output", type=Path, default=Path("data/human_prototype/interim/motifs_chromvar.tsv"))
    parser.add_argument("--log_path", type=Path, default=Path("logs/human_prototype/compute_chromvar.log"))
    args = parser.parse_args()

    configure_logging(args.log_path)
    meta = pd.read_csv(args.metadata, sep="\t")

    # Pick the target nuclei: all QC nuclei, or QC nuclei of one label.
    cell_selection = meta["pass_qc"] if args.cell_label == "all" else \
        (meta["pass_qc"] & (meta[args.label_column] == args.cell_label))
    keep_barcodes = set(meta.loc[cell_selection, "barcode"])
    logger.info(f"Target {args.label_column}={args.cell_label} cells: {len(keep_barcodes)}")

    adata = build_adata(args.snap, keep_barcodes, args.min_cells_per_peak, args.chroms)
    deviation_df = run_chromvar(adata, str(args.genome), args.jaspar_release, args.bg_iterations)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    deviation_df.to_csv(args.output, sep="\t")
    logger.info(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
