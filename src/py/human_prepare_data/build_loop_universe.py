# human_prepare_data/build_loop_universe.py

"""
Build per-condition human loop sets from the three Rahman et al. 2023 processed
loop BED files (fetal / glia / neuron).

Each condition C defines its own labelled loop set on native (per-condition)
coordinates. Positives are always condition C's own loops (label 1); how the
negatives (label 0) are drawn is chosen with --negatives:

  other_conditions = the OTHER conditions' loops whose Hi-C rectangle does not
                     coincide with ANY actual C loop rectangle
  n2               = new pairs of C's own anchors (Cupriak's "new_pairs_of_anchors")

Mode other_conditions: disjoint-rectangles criterion (pairwise, against real loops)
------------------------------------------------------------
A loop is a rectangle on the Hi-C plane: [x1, x2] x [y1, y2], the left anchor on
one axis and the right anchor on the other. Two axis-aligned rectangles overlap
iff they overlap on BOTH axes (both anchors overlap). The three conditions share
almost all of their anchors -- what differs is how those anchors are PAIRED into
loops -- so overlap must be tested against each condition's actual loops, not
against the union of its anchor territory (a candidate can hit two different C
anchors that are never paired in C).

Hence, per condition C:
  - an other-condition loop is EXCLUDED (label NaN) when its rectangle overlaps at
    least one real C loop rectangle: both its anchors overlap the two anchors of
    one and the same C loop, i.e. it reproduces an existing C pairing and must not
    be trained against as a C negative;
  - otherwise it is a clean NEGATIVE (label 0): its anchors may individually recur
    in C, but that specific pairing is absent from C.

Mode n2: random re-pairing of the condition's own anchors
--------------------------------------------------------
A port of new_pairs_of_anchors (negative type n2) from Cupriak's repository
(github.com/ola-cupriak/predicting-the-formation-of-chromatin-loops-using-genomic-data,
pipelines/data_preprocessing/nodes.py), per condition C:
  1. pool the left and right anchors of C's loops into one set of distinct anchors;
  2. pair every two anchors on the same chromosome whose intervals do not
     overlap (left end < right start), the left anchor first;
  3. drop the pairs that are exactly one of C's loops;
  4. draw int(neg_pos_ratio * #C loops) of the remaining pairs, uniformly
     without replacement (seeded);
  5. drop the drawn pairs whose rectangle overlaps a real C loop rectangle (the
     same test as above). Cupriak applies this filter after the draw, so the
     final ratio ends slightly below neg_pos_ratio.
Anchors keep their native Rahman intervals. Every real loop is a positive of its
own condition only, and each condition's n2 pairs are negatives of that condition
only (label NaN elsewhere): in this mode no loop of another condition is a negative.

All three condition sets are emitted as one table of native loops (no coordinate
merging) with one label column per condition. Downstream (anchor openness, motif
enrichment, training) reads label_<C> for condition C and drops the excluded
(NaN) rows.

Inputs:
  - data/human_prototype/raw/{fetal,glia,neuron}_loop.bed.gz
Outputs:
  - data/human_prototype/interim/<negatives>/loop_universe.tsv
    (chr1 x1 x2 chr2 y1 y2 color loop_id source label_fetal label_glia label_neuron;
    source is the condition of a real loop, or n2_<C> for C's drawn negatives)
"""

import argparse
import gzip
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from pybedtools import BedTool

from core.log import configure_logging
from human_prepare_data.constants import CONDITIONS, NEGATIVE_MODES

logger = logging.getLogger(__name__)

# Second anchor is encoded in the BED name column as "chrN:start-end".
_ANCHOR2_RE = re.compile(r"^(chr[\w]+):(\d+)-(\d+)")


def read_loop_bed(path: Path) -> pd.DataFrame:
    """Parse one condition's loop BED into canonical, de-duplicated loop rows.

    Anchors are ordered so the left anchor (smaller start) is (x1, x2) and the
    right anchor is (y1, y2); only intra-chromosomal loops are kept. Rahman lists
    every loop twice (both anchor orientations), which canonical ordering makes
    identical, so exact duplicates are dropped to get one row per distinct loop.
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

    loops = (pd.DataFrame(rows, columns=["chr", "x1", "x2", "y1", "y2"])
             .drop_duplicates(ignore_index=True))
    logger.info(f"{path.name}: {len(loops)} distinct loops")

    return loops


def _loops_to_bedpe(loops: pd.DataFrame, include_id: bool) -> BedTool:
    """Encode loops as a BEDPE (chrom1 start1 end1 chrom2 start2 end2 [name]).

    Each loop becomes one paired-end record: anchor 1 = (x1, x2), anchor 2 =
    (y1, y2). The optional name column carries loop_id so matches can be traced
    back to the candidate loop.
    """
    cols = {0: loops["chr"].to_numpy(), 1: loops["x1"].to_numpy(), 2: loops["x2"].to_numpy(),
            3: loops["chr"].to_numpy(), 4: loops["y1"].to_numpy(), 5: loops["y2"].to_numpy()}
    if include_id:
        cols[6] = loops["loop_id"].to_numpy()
    return BedTool.from_dataframe(pd.DataFrame(cols))


def loops_overlapping_condition(candidates: pd.DataFrame, c_loops: pd.DataFrame) -> set[str]:
    """loop_ids of candidate loops whose rectangle overlaps >=1 real C loop rectangle.

    This is the every-loop-vs-every-loop 2-D rectangle-intersection join, done the
    output-sensitive way rather than as the O(candidates x C-loops) product:

      * each loop is a BEDPE paired-end record (two intervals = the two axes);
      * bedtools `pairToPair -type both` reports a candidate<->C pair only when
        BOTH ends overlap -- exactly a 2-D rectangle intersection;
      * internally it indexes the C loops (binned interval tree) and sweeps the
        candidates, so for each candidate it only visits the C loops that already
        overlap on one axis (the small "active set" of loops spanning that locus,
        bounded by local loop density), then checks the other axis.

    A candidate appears in the output iff it reproduces some real C pairing; those
    ids are excluded as C negatives.
    """
    a = _loops_to_bedpe(candidates, include_id=True)
    b = _loops_to_bedpe(c_loops, include_id=False)

    hits = a.pair_to_pair(b=b.fn, type="both")
    if len(hits) == 0:
        return set()

    matched = hits.to_dataframe(disable_auto_names=True, header=None)[6]  # candidate loop_id
    return set(matched.unique())


def assign_labels(loops: pd.DataFrame, per_cond: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Add a label_<C> column (1 positive / 0 negative / NaN excluded) per condition."""
    for cond in CONDITIONS:
        is_source = loops["source"] == cond
        others = loops.loc[~is_source]
        excluded_ids = loops_overlapping_condition(others, per_cond[cond])

        is_excluded = (~is_source) & loops["loop_id"].isin(excluded_ids)   # reproduces a C pairing
        is_negative = (~is_source) & (~is_excluded)                        # pairing absent from C
        label = pd.Series(np.nan, index=loops.index, dtype="float")
        label[is_source] = 1.0                                             # C's own loops
        label[is_negative] = 0.0                                           # disjoint-rectangle negatives

        loops[f"label_{cond}"] = label
        n_pos, n_neg, n_excl = int(is_source.sum()), int(is_negative.sum()), int(is_excluded.sum())
        logger.info(f"  {cond}: {n_pos} positives, {n_neg} negatives, {n_excl} excluded "
                    f"(rectangle coincides with a {cond} loop) | "
                    f"{100 * n_pos / (n_pos + n_neg):.1f}% positive")

    return loops


def condition_anchors(c_loops: pd.DataFrame) -> pd.DataFrame:
    """C's distinct anchors (left and right pooled), sorted by (chr, start, end)."""
    sides = [c_loops[["chr", s, e]].set_axis(["chr", "start", "end"], axis=1)
             for s, e in (("x1", "x2"), ("y1", "y2"))]
    return (pd.concat(sides).drop_duplicates()
            .sort_values(["chr", "start", "end"], ignore_index=True))


def candidate_anchor_pairs(anchors: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(left, right) row indices into `anchors` of every non-overlapping same-chromosome pair.

    Each unordered pair appears once: anchors are sorted by start, so the earlier
    row is the left anchor, and the pair is kept iff left end < right start
    (Cupriak's x_end < y_start).
    """
    starts, ends = anchors["start"].to_numpy(), anchors["end"].to_numpy()

    left, right = [], []
    for rows in anchors.groupby("chr", sort=True).indices.values():
        i, j = np.triu_indices(len(rows), k=1)
        i, j = rows[i], rows[j]
        keep = ends[i] < starts[j]
        left.append(i[keep].astype(np.int32))
        right.append(j[keep].astype(np.int32))

    return np.concatenate(left), np.concatenate(right)


def n2_negatives(c_loops: pd.DataFrame, neg_pos_ratio: float, random_state: int,
                 cond: str) -> pd.DataFrame:
    """Condition C's n2 negatives: random new pairs of C's own anchors (see module docstring)."""
    anchors = condition_anchors(c_loops)
    left, right = candidate_anchor_pairs(anchors)

    # Pairs are keyed as left * n + right; drop the ones that are one of C's loops.
    n = len(anchors)
    anchor_index = pd.MultiIndex.from_frame(anchors)
    pos_left = anchor_index.get_indexer(pd.MultiIndex.from_frame(c_loops[["chr", "x1", "x2"]]))
    pos_right = anchor_index.get_indexer(pd.MultiIndex.from_frame(c_loops[["chr", "y1", "y2"]]))
    is_loop = np.isin(left.astype(np.int64) * n + right,
                      pos_left.astype(np.int64) * n + pos_right)
    left, right = left[~is_loop], right[~is_loop]

    # Uniform draw without replacement: pandas DataFrame.sample(random_state=int)
    # draws exactly this way (legacy RandomState.choice).
    n_draw = int(neg_pos_ratio * len(c_loops))
    if n_draw > len(left):
        raise ValueError(f"{cond}: {n_draw} n2 negatives requested but only {len(left)} anchor pairs exist")
    drawn = np.sort(np.random.RandomState(random_state).choice(len(left), size=n_draw, replace=False))
    left, right = left[drawn], right[drawn]

    negatives = pd.DataFrame({
        "chr": anchors["chr"].to_numpy()[left],
        "x1": anchors["start"].to_numpy()[left], "x2": anchors["end"].to_numpy()[left],
        "y1": anchors["start"].to_numpy()[right], "y2": anchors["end"].to_numpy()[right],
    })
    negatives["loop_id"] = [f"tmp{i}" for i in range(len(negatives))]
    overlapping = negatives["loop_id"].isin(loops_overlapping_condition(negatives, c_loops))
    negatives = negatives.loc[~overlapping].drop(columns="loop_id").reset_index(drop=True)

    logger.info(f"  {cond}: {n} distinct anchors -> {len(is_loop)} non-overlapping pairs, "
                f"{int(is_loop.sum())} of them real loops (dropped) -> drew {n_draw} "
                f"(ratio {neg_pos_ratio}) -> {int(overlapping.sum())} overlap a {cond} loop "
                f"rectangle (dropped) -> {len(negatives)} negatives")

    return negatives


def add_n2_negatives(loops: pd.DataFrame, per_cond: dict[str, pd.DataFrame],
                     neg_pos_ratio: float, random_state: int) -> pd.DataFrame:
    """Label each real loop for its own condition only and append every condition's n2 negatives."""
    for cond in CONDITIONS:
        loops[f"label_{cond}"] = np.where(loops["source"] == cond, 1.0, np.nan)

    blocks = [loops]
    for cond in CONDITIONS:
        negatives = n2_negatives(per_cond[cond], neg_pos_ratio, random_state, cond)
        negatives["source"] = f"n2_{cond}"
        for c in CONDITIONS:
            negatives[f"label_{c}"] = 0.0 if c == cond else np.nan
        blocks.append(negatives)

    # Real loops keep their ids (the same as in mode other_conditions); negatives continue the numbering.
    n_real = len(loops)
    loops = pd.concat(blocks, ignore_index=True)
    loops.loc[n_real:, "loop_id"] = [f"L{i}" for i in range(n_real + 1, len(loops) + 1)]

    for cond in CONDITIONS:
        n_pos, n_neg = int((loops[f"label_{cond}"] == 1).sum()), int((loops[f"label_{cond}"] == 0).sum())
        logger.info(f"  {cond}: {n_pos} positives, {n_neg} n2 negatives | "
                    f"{100 * n_pos / (n_pos + n_neg):.1f}% positive")

    return loops


def build_loop_sets(raw_dir: Path, negatives: str = "other_conditions",
                    neg_pos_ratio: float = 5, random_state: int = 42) -> pd.DataFrame:
    """Pool the three conditions' native loops and label each per condition."""
    per_cond = {c: read_loop_bed(raw_dir / f"{c}_loop.bed.gz") for c in CONDITIONS}

    loops = pd.concat([df.assign(source=c) for c, df in per_cond.items()], ignore_index=True)
    loops.insert(0, "loop_id", [f"L{i}" for i in range(1, len(loops) + 1)])
    logger.info(f"Native loops pooled across conditions (no merging): {len(loops)}")

    logger.info(f"Negatives: {negatives}")
    if negatives == "n2":
        loops = add_n2_negatives(loops, per_cond, neg_pos_ratio, random_state)
    else:
        loops = assign_labels(loops, per_cond)

    loops = loops.rename(columns={"chr": "chr1"})
    loops.insert(4, "chr2", loops["chr1"])
    loops["color"] = "0,0,0"

    return loops[["chr1", "x1", "x2", "chr2", "y1", "y2", "color", "loop_id", "source",
                  *(f"label_{c}" for c in CONDITIONS)]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-condition human loop sets (positives + negatives).")
    parser.add_argument("--raw_dir", type=Path, default=Path("data/human_prototype/raw"))
    parser.add_argument("--negatives", default="other_conditions", choices=NEGATIVE_MODES)
    parser.add_argument("--neg_pos_ratio", type=float, default=5,
                        help="n2 only: negatives drawn per positive (before the rectangle-overlap filter).")
    parser.add_argument("--random_state", type=int, default=42, help="n2 only: seed of the negative draw.")
    parser.add_argument("--output", type=Path, default=Path("data/human_prototype/interim/other_conditions/loop_universe.tsv"))
    parser.add_argument("--log_path", type=Path, default=Path("logs/human_prototype/other_conditions/build_loop_universe.log"))
    args = parser.parse_args()

    configure_logging(args.log_path)
    loops = build_loop_sets(args.raw_dir, args.negatives, args.neg_pos_ratio, args.random_state)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    loops.to_csv(args.output, sep="\t", index=False)
    logger.info(f"Wrote {args.output} ({len(loops)} native loops)")


if __name__ == "__main__":
    main()
