# human_prepare_data/annotate_cells.py

"""
Attach the published Li et al. 2023 atlas annotation to the SNAP nuclei.

Every nucleus is labelled from the atlas's per-nucleus metatable (Table S3,
`hba.whole.final.meta.txt`) by joining on (sample, barcode): its coarse class
(GLUT / GABA / NonN), its subclass (one of the 42), and its cell type (one of
the 107). The coarse `label` column collapses these to neuron / non_neuron for
the cell-set interface; the atlas columns are kept for finer subsets. Nuclei
absent from the atlas (dropped by the atlas's own QC / doublet removal) are
labelled `unannotated`.

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
from human_prepare_data.constants import ATLAS_COLUMNS, NEURONAL_CLASSES, NON_NEURONAL_CLASS
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


def annotate(snap_path: Path, annotation_path: Path, sample: str, min_fragments: int) -> pd.DataFrame:
    """Return per-nucleus QC flag and atlas class / subclass / cell-type labels, in snap order."""
    with h5py.File(snap_path, "r") as f:
        barcodes = load_barcodes(f)
        frag_counts = load_fragment_counts(f)

    atlas = load_atlas_annotation(annotation_path, sample)

    meta = pd.DataFrame({"barcode": barcodes, "n_fragments": frag_counts})
    meta["pass_qc"] = meta["n_fragments"] >= min_fragments

    # Join the atlas columns onto our nuclei; anything unmatched is "unannotated".
    for col in ("cellclass", "subclass", "celltype"):
        meta[col] = meta["barcode"].map(atlas[col]).fillna("unannotated")

    # Collapse the atlas coarse class into the two cell-set labels.
    class_to_label = {cls: "neuron" for cls in NEURONAL_CLASSES}
    class_to_label[NON_NEURONAL_CLASS] = "non_neuron"
    meta["label"] = meta["cellclass"].map(class_to_label).fillna("unannotated")

    n_annotated = int((meta["cellclass"] != "unannotated").sum())
    logger.info(f"QC: {int(meta['pass_qc'].sum())} / {len(meta)} nuclei pass >= {min_fragments} unique fragments")
    logger.info(f"Atlas match: {n_annotated} / {len(meta)} snap nuclei annotated")
    logger.info(f"cellclass among QC nuclei: {meta.loc[meta['pass_qc'], 'cellclass'].value_counts().to_dict()}")
    logger.info(f"subclass among QC nuclei: {meta.loc[meta['pass_qc'], 'subclass'].value_counts().to_dict()}")
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
