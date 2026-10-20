# human_prepare_data/annotate_cells.py

"""
Attach the published Li et al. 2023 atlas annotation to the SNAP nuclei.

Every nucleus is labelled from the atlas's per-nucleus metatable (Table S3,
`hba.whole.final.meta.txt`) by joining on (sample, barcode): its coarse class
(GLUT / GABA / NonN), its subclass (one of the 42), and its cell type (one of
the 107). Nuclei absent from the atlas (dropped by the atlas's own QC /
doublet removal) are labelled `unannotated`, with `annotated` = False.

The `in_analysis` column (passes QC and is annotated) defines the analysis
cells: chromVAR, anchor openness and Stage 0 all use exactly these nuclei, so
every cell can be assigned to a group of an atlas-based cell partition.

Inputs:
  - a GSE244618 .snap file
  - the atlas single-nucleus metatable (CATlas Table S3, gzipped)
Outputs:
  - data/human_prototype/interim/cell_metadata.tsv
"""

import argparse
import logging
from pathlib import Path

import h5py
import pandas as pd

from core.log import configure_logging
from human_prepare_data.constants import ATLAS_COLUMNS, UNANNOTATED
from human_prepare_data.snap_io import load_barcodes, load_fragment_counts

logger = logging.getLogger(__name__)


def load_atlas_annotation(annotation_path: Path, sample: str) -> pd.DataFrame:
    """Read the atlas metatable and return the rows for one library, barcode-indexed."""
    atlas = pd.read_csv(annotation_path, sep="\t", usecols=ATLAS_COLUMNS, dtype=str)

    rows = atlas[atlas["sample"] == sample]
    if rows.empty:
        raise ValueError(f"No atlas rows for sample {sample!r}; check --sample against Table S3.")

    logger.info(f"Atlas rows for sample {sample}: {len(rows)} (of {len(atlas)} total nuclei)")
    return rows.set_index("barcode")


def log_dropped_unannotated(meta: pd.DataFrame) -> None:
    """Log how many QC-passing nuclei are left out of the analysis for lacking an atlas label."""
    n_qc = int(meta["pass_qc"].sum())
    n_dropped = int((meta["pass_qc"] & ~meta["annotated"]).sum())
    logger.info(f"Dropping {n_dropped} unannotated QC nuclei ({100 * n_dropped / n_qc:.1f}% of "
                f"{n_qc} QC-passing nuclei); {n_qc - n_dropped} analysis cells remain")


def annotate(snap_path: Path, annotation_path: Path, sample: str, min_fragments: int) -> pd.DataFrame:
    """Return per-nucleus QC / annotation / analysis flags and atlas labels, in snap order."""
    with h5py.File(snap_path, "r") as f:
        barcodes = load_barcodes(f)
        frag_counts = load_fragment_counts(f)

    atlas = load_atlas_annotation(annotation_path, sample)

    meta = pd.DataFrame({"barcode": barcodes, "n_fragments": frag_counts})
    meta["pass_qc"] = meta["n_fragments"] >= min_fragments

    # Join the atlas columns onto our nuclei; anything unmatched is "unannotated".
    meta["annotated"] = meta["barcode"].isin(atlas.index)
    for col in ("cellclass", "subclass", "celltype"):
        meta[col] = meta["barcode"].map(atlas[col]).fillna(UNANNOTATED)

    # Analysis cells: QC-passing nuclei that carry an atlas label.
    meta["in_analysis"] = meta["pass_qc"] & meta["annotated"]

    logger.info(f"QC: {int(meta['pass_qc'].sum())} / {len(meta)} nuclei pass >= {min_fragments} unique fragments")
    logger.info(f"Atlas match: {int(meta['annotated'].sum())} / {len(meta)} snap nuclei annotated")
    log_dropped_unannotated(meta)
    logger.info(f"cellclass among analysis cells: {meta.loc[meta['in_analysis'], 'cellclass'].value_counts().to_dict()}")
    logger.info(f"subclass among analysis cells: {meta.loc[meta['in_analysis'], 'subclass'].value_counts().to_dict()}")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach the Li 2023 atlas annotation (Table S3) to the SNAP nuclei.")
    parser.add_argument("--snap", type=Path, required=True)
    parser.add_argument("--annotation", type=Path,
                        default=Path("data/human_prototype/annotation/TableS3_single_nuclei_metatable.gz"),
                        help="CATlas Table S3 gzipped metatable (hba.whole.final.meta.txt).")
    parser.add_argument("--sample", default="MM_417",
                        help="Atlas library id for this SNAP (e.g. MM_417).")
    parser.add_argument("--min_fragments", type=int, default=1000, help="Unique-fragment QC threshold.")
    parser.add_argument("--output", type=Path, default=Path("data/human_prototype/interim/cell_metadata.tsv"))
    parser.add_argument("--log_path", type=Path, default=Path("logs/human_prototype/annotate_cells.log"))
    args = parser.parse_args()

    configure_logging(args.log_path)
    meta = annotate(args.snap, args.annotation, args.sample, args.min_fragments)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(args.output, sep="\t", index=False)
    logger.info(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
