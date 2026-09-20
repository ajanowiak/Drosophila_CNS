# human_prepare_data/build_loop_universe.py

"""
Build a unified human loop universe with per-condition binary presence from the
three Rahman et al. 2023 processed loop BED files (fetal / glia / neuron),
mirroring the Drosophila long_and_short_range_loops_D_mel.tsv layout.

Loops are matched across (and within) conditions by reciprocal anchor overlap
and grouped into connected components; each component is one universe loop.

Inputs:
  - data/human_prototype/raw/{fetal,glia,neuron}_loop.bed.gz
Outputs:
  - data/human_prototype/interim/loop_universe.tsv
    (chr1 x1 x2 chr2 y1 y2 color loop_id Human_fetal Human_glia Human_neuron)
"""

import argparse
import gzip
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from core.log import configure_logging
from human_prepare_data.constants import CONDITIONS

logger = logging.getLogger(__name__)

# Second anchor is encoded in the BED name column as "chrN:start-end".
_ANCHOR2_RE = re.compile(r"^(chr[\w]+):(\d+)-(\d+)")


def read_loop_bed(path: Path) -> pd.DataFrame:
    """Parse one condition's loop BED into canonical two-anchor rows.

    Anchors are ordered so the left anchor (smaller start) is (x1, x2) and the
    right anchor is (y1, y2); only intra-chromosomal loops are kept.
    """
    rows = []
    with gzip.open(path, "rt") as fh:
        for line in fh:

            fields = line.rstrip("\n").split("\t")
            if len(fields) < 4:
                continue

            # First anchor is columns 0-2; second is packed into the name column.
            match = _ANCHOR2_RE.match(fields[3])
            if not match or match.group(1) != fields[0]:
                continue

            (x1, x2), (y1, y2) = sorted([(int(fields[1]), int(fields[2])),
                                         (int(match.group(2)), int(match.group(3)))])
            rows.append((fields[0], x1, x2, y1, y2))

    loops = pd.DataFrame(rows, columns=["chr", "x1", "x2", "y1", "y2"])
    logger.info(f"{path.name}: {len(loops)} loops")

    return loops


def overlap_edges(df: pd.DataFrame, frac: float) -> tuple[list[int], list[int]]:
    """Edges (by df index) between loops whose BOTH anchors reciprocally overlap.

    Two loops are linked when their left anchors and their right anchors each
    overlap by at least `frac` of the shorter anchor. Loops are swept in
    left-anchor order, so each is only compared against later loops whose left
    anchor still overlaps.
    """
    sorted_loops = df.sort_values("x1")
    idx = sorted_loops.index.to_numpy()
    x1, x2 = sorted_loops["x1"].to_numpy(), sorted_loops["x2"].to_numpy()
    y1, y2 = sorted_loops["y1"].to_numpy(), sorted_loops["y2"].to_numpy()

    src, dst = [], []
    for i in range(len(sorted_loops)):
        for j in range(i + 1, len(sorted_loops)):

            # Sorted by left start, so once j's left anchor starts past i's, no
            # later loop can overlap i's left anchor either.
            if x1[j] >= x2[i]:
                break

            left_overlap = min(x2[i], x2[j]) - max(x1[i], x1[j])
            right_overlap = min(y2[i], y2[j]) - max(y1[i], y1[j])
            if left_overlap <= 0 or right_overlap <= 0:
                continue

            if left_overlap >= frac * min(x2[i] - x1[i], x2[j] - x1[j]) and \
               right_overlap >= frac * min(y2[i] - y1[i], y2[j] - y1[j]):
                src.append(idx[i])
                dst.append(idx[j])

    return src, dst


def build_universe(raw_dir: Path, frac: float) -> pd.DataFrame:
    """Pool the three conditions' loops and cluster them into a shared universe."""
    pooled_loops = pd.concat(
        [read_loop_bed(raw_dir / f"{c}_loop.bed.gz").assign(cond=c) for c in CONDITIONS],
        ignore_index=True,
    )
    logger.info(f"Pooled loop calls across conditions: {len(pooled_loops)}")

    # Overlap is only ever within a chromosome, so build edges chromosome by
    # chromosome and offset nothing (indices are the pooled-frame indices).
    src, dst = [], []
    for _, chrom_loops in pooled_loops.groupby("chr"):
        chrom_src, chrom_dst = overlap_edges(chrom_loops, frac)
        src.extend(chrom_src)
        dst.extend(chrom_dst)

    # Each connected component of the overlap graph becomes one universe loop.
    n_loops = len(pooled_loops)
    graph = coo_matrix((np.ones(len(src)), (src, dst)), shape=(n_loops, n_loops))
    _, pooled_loops["cluster"] = connected_components(graph, directed=False)
    logger.info(f"Universe loops after overlap clustering: {pooled_loops['cluster'].nunique()}")

    # Representative coordinates: the earliest-starting loop in each cluster.
    representatives = (pooled_loops.sort_values("x1")
                       .groupby("cluster").first()[["chr", "x1", "x2", "y1", "y2"]])
    # A cluster is "present" in a condition if any of its member loops came from it.
    presence = (pooled_loops.groupby(["cluster", "cond"]).size().unstack(fill_value=0) > 0).astype(int)

    universe = representatives.reset_index(drop=True)
    universe.insert(0, "loop_id", [f"L{i}" for i in range(1, len(universe) + 1)])
    universe = universe.rename(columns={"chr": "chr1"})
    universe.insert(3, "chr2", universe["chr1"])
    universe["color"] = "0,0,0"

    for cond in CONDITIONS:
        universe[f"Human_{cond}"] = presence[cond].to_numpy() if cond in presence else 0
        n_pos = int(universe[f"Human_{cond}"].sum())
        logger.info(f"  {cond} positives: {n_pos} / {len(universe)} ({100 * universe[f'Human_{cond}'].mean():.1f}%)")

    return universe[["chr1", "x1", "x2", "chr2", "y1", "y2", "color", "loop_id",
                     *(f"Human_{c}" for c in CONDITIONS)]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the human loop universe with per-condition presence.")
    parser.add_argument("--raw_dir", type=Path, default=Path("data/human_prototype/raw"))
    parser.add_argument("--frac", type=float, default=0.5,
                        help="Min reciprocal overlap fraction on both anchors to link two loops.")
    parser.add_argument("--output", type=Path, default=Path("data/human_prototype/interim/loop_universe.tsv"))
    parser.add_argument("--log_path", type=Path, default=Path("logs/human_prototype/build_loop_universe.log"))
    args = parser.parse_args()

    configure_logging(args.log_path)
    universe = build_universe(args.raw_dir, args.frac)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    universe.to_csv(args.output, sep="\t", index=False)
    logger.info(f"Wrote {args.output} ({len(universe)} loops)")


if __name__ == "__main__":
    main()
