# human_prepare_data/run_stage0_human.py

"""
Human Stage 0 driver: motif-enrichment matrix + per-condition label vectors.

Reuses the Drosophila enrichment math (compute_group_enrichment) unchanged; the
only human glue is aligning the loop-activity and chromVAR matrices on shared
cells and writing each condition's label vector from the loop table. Enrichment
is loop-local, so the one enrichment matrix per cell set doubles as every
condition's feature matrix - each y_<condition> just selects that condition's
rows (positives + clean negatives; excluded loops are dropped).

Inputs:
  - loops_activity.tsv, motifs_chromvar.tsv, loop_universe.tsv
Outputs (results/human_prototype/prepare_data/):
  - motif_enrichment.csv, count11.csv, n_cells.txt, y_<condition>.csv
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from core.log import configure_logging
from human_prepare_data.constants import CONDITIONS
from prepare_data.compute_motif_enrichment import compute_group_enrichment

logger = logging.getLogger(__name__)


def align_cells(loops_df: pd.DataFrame, motifs_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict both matrices to their shared cells, in the same order."""
    shared_cells = loops_df.columns.intersection(motifs_df.columns)
    logger.info(f"Shared cells: {len(shared_cells)} (loops {loops_df.shape[1]}, motifs {motifs_df.shape[1]})")

    if len(shared_cells) == 0:
        raise ValueError("No shared cells between loop-activity and chromVAR matrices.")

    return loops_df[shared_cells], motifs_df[shared_cells]


def main() -> None:
    parser = argparse.ArgumentParser(description="Human Stage 0: motif enrichment + loop presence vectors.")
    parser.add_argument("--loops", type=Path, default=Path("data/human_prototype/interim/loops_activity.tsv"))
    parser.add_argument("--motifs", type=Path, default=Path("data/human_prototype/interim/motifs_chromvar.tsv"))
    parser.add_argument("--loop_table", type=Path, default=Path("data/human_prototype/interim/loop_universe.tsv"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/human_prototype/prepare_data"))
    parser.add_argument("--log_path", type=Path, default=Path("logs/human_prototype/stage0.log"))
    args = parser.parse_args()

    configure_logging(args.log_path)
    loops_df = pd.read_csv(args.loops, sep="\t", index_col=0)
    motifs_df = pd.read_csv(args.motifs, sep="\t", index_col=0)

    loops_df, motifs_df = align_cells(loops_df, motifs_df)
    loops_df = loops_df.astype("int8")   # activity codes are 0/1/10/11; keeps big matrices small

    logger.info("Computing motif enrichment (reusing compute_group_enrichment) ...")
    enrichment_df, count_11 = compute_group_enrichment(loops_df, motifs_df)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    enrichment_df.to_csv(args.output_dir / "motif_enrichment.csv")
    logger.info(f"Wrote motif_enrichment.csv  shape={enrichment_df.shape}")
    pd.Series(count_11, index=enrichment_df.index, name="count11").to_csv(args.output_dir / "count11.csv")

    # Record how many cells defined this cell set's openness, for Stage 1 titles.
    (args.output_dir / "n_cells.txt").write_text(f"{loops_df.shape[1]}\n")

    loop_table = pd.read_csv(args.loop_table, sep="\t").set_index("loop_id")
    for cond in CONDITIONS:
        # Each condition has its own loop set: positives (label 1) + clean
        # negatives (label 0). Loops excluded for this condition are NaN, so
        # dropping them makes y_<cond> restrict the shared enrichment matrix to
        # exactly this condition's loops (its own feature matrix, by row).
        y = loop_table.loc[enrichment_df.index, f"label_{cond}"].dropna().astype(int)
        y.to_csv(args.output_dir / f"y_{cond}.csv")
        logger.info(f"  y_{cond}: {int(y.sum())} positives / {int((y == 0).sum())} negatives "
                    f"({len(y)} loops)")

    logger.info("Stage 0 done.")


if __name__ == "__main__":
    main()
