# human_prepare_data/run_stage0_pooled.py

"""
Stage 0 on cells pooled from several libraries: motif enrichment + label vectors.

Adds up the per-loop sums and counts of the pooled libraries (compute_loop_sums.py)
and turns them into enrichment, mean over "1-1" cells - mean over the other cells,
exactly as if the libraries' cells formed one library (chromVAR scores stay per
library). Libraries can be split into groups, e.g. by brain region: each group's
cells are pooled separately and the groups' matrices are stacked side by side, one
feature per motif x group, named "<motif>__<group>". Without groups all libraries
form one pool and features are plain motif names.

Prototype step of the pooled-cell models (pfc_pools.smk). The outputs have the
layout of run_stage0_human.py, so run_stage1_human.py trains on them unchanged.

Inputs:
  - loop_sums.npz of each pooled library, <negatives>/loop_universe.tsv
Outputs (--output_dir):
  - motif_enrichment.csv, count11.csv (loop x group), n_cells.tsv (cells per group),
    y_<condition>.csv
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from core.log import configure_logging
from human_prepare_data.constants import GROUP_SEPARATOR, POOLED_GROUP
from human_prepare_data.run_stage0_human import write_label_vectors

logger = logging.getLogger(__name__)

ADDITIVE_ARRAYS = ["sum_11", "sum_other", "count_11"]


def load_sums(path: Path, loop_ids: np.ndarray | None, motifs: np.ndarray | None) -> dict:
    """One library's loop sums; checks its loops and motifs match the libraries read before it."""
    with np.load(path) as sums:
        library = {name: sums[name] for name in sums.files}
    if loop_ids is not None and not np.array_equal(library["loop_ids"], loop_ids):
        raise ValueError(f"{path} has other loops than the libraries before it")
    if motifs is not None and not np.array_equal(library["motifs"], motifs):
        raise ValueError(f"{path} has other motifs than the libraries before it")
    return library


def pool_group(paths: list[Path]) -> dict:
    """Add up the sums, "1-1" counts and cell numbers of one group's libraries."""
    pooled = load_sums(paths[0], None, None)
    logger.info(f"  added {paths[0]} ({int(pooled['n_cells'])} cells)")
    for path in paths[1:]:
        library = load_sums(path, pooled["loop_ids"], pooled["motifs"])
        for name in ADDITIVE_ARRAYS + ["n_cells"]:
            pooled[name] += library[name]
        logger.info(f"  added {path} ({int(library['n_cells'])} cells)")
    return pooled


def pooled_enrichment(pooled: dict) -> np.ndarray:
    """Mean over "1-1" cells minus mean over the other cells; NaN where a group is empty."""
    count_11 = pooled["count_11"][:, None]
    count_other = int(pooled["n_cells"]) - count_11
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_11 = np.where(count_11 > 0, pooled["sum_11"] / count_11, np.nan)
        mean_other = np.where(count_other > 0, pooled["sum_other"] / count_other, np.nan)
    return mean_11 - mean_other


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 0 on cells pooled from several libraries.")
    parser.add_argument("--loop_sums", type=Path, nargs="+", required=True, help="loop_sums.npz of each library.")
    parser.add_argument("--groups", nargs="+", default=None,
                        help="Group of each --loop_sums library (same order); omit to pool all libraries together.")
    parser.add_argument("--loop_table", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--log_path", type=Path, required=True)
    args = parser.parse_args()

    if args.groups is not None and len(args.groups) != len(args.loop_sums):
        parser.error("--groups needs one group per --loop_sums file")

    configure_logging(args.log_path)
    grouped = args.groups is not None
    groups = pd.Series(args.groups if grouped else [POOLED_GROUP] * len(args.loop_sums), index=args.loop_sums)

    enrichment_blocks, count11_by_group, n_cells_by_group = [], {}, {}
    for group in sorted(groups.unique()):
        logger.info(f"Group {group}:")
        pooled = pool_group(list(groups.index[groups == group]))
        features = ([f"{motif}{GROUP_SEPARATOR}{group}" for motif in pooled["motifs"]] if grouped
                    else list(pooled["motifs"]))
        enrichment_blocks.append(pd.DataFrame(pooled_enrichment(pooled), index=pooled["loop_ids"], columns=features))
        count11_by_group[group] = pd.Series(pooled["count_11"], index=pooled["loop_ids"])
        n_cells_by_group[group] = int(pooled["n_cells"])
        logger.info(f"  {n_cells_by_group[group]} cells, loops with >=1 both-open cell: "
                    f"{int((pooled['count_11'] > 0).sum())} / {len(pooled['loop_ids'])}")
        del pooled

    enrichment = pd.concat(enrichment_blocks, axis=1).rename_axis("loop_id")
    if not all(block.index.equals(enrichment.index) for block in enrichment_blocks):
        raise ValueError("groups have different loops")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    enrichment.to_csv(args.output_dir / "motif_enrichment.csv")
    logger.info(f"Wrote motif_enrichment.csv  shape={enrichment.shape}")
    pd.DataFrame(count11_by_group).rename_axis("loop_id").to_csv(args.output_dir / "count11.csv")
    (pd.Series(n_cells_by_group).sort_index().rename_axis("group").rename("n_cells")
     .to_csv(args.output_dir / "n_cells.tsv", sep="\t"))

    loop_table = pd.read_csv(args.loop_table, sep="\t").set_index("loop_id")
    write_label_vectors(loop_table, enrichment.index, args.output_dir)
    logger.info("Pooled Stage 0 done.")


if __name__ == "__main__":
    main()
