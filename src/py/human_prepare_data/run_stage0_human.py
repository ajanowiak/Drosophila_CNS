# human_prepare_data/run_stage0_human.py

"""
Human Stage 0 driver: motif-enrichment matrix + per-condition label vectors.

Builds the loop x cell activity codes of one loop universe from the shared
anchor openness matrix, then reuses the Drosophila enrichment math
(compute_group_enrichment) unchanged. One run produces one feature set:

  - without a cell partition, enrichment contrasts the "1-1" cells with all
    other analysis cells: one feature per motif;
  - with a partition (e.g. the atlas cellclass), the "1-1" and "other"
    populations are defined within each group's cells separately, and the
    groups' matrices are stacked side by side: one feature per motif x group,
    named "<motif>__<group>". Every analysis cell must belong to exactly one group.

Enrichment is loop-local, so the one enrichment matrix doubles as every
condition's feature matrix - each y_<condition> just selects that condition's
rows (positives + clean negatives; excluded loops are dropped).

Inputs:
  - anchor_openness.tsv, motifs_chromvar.tsv, <negatives>/loop_universe.tsv
  - optionally a cell partition table (barcode + group column)
Outputs (results/human_prototype/<negatives>/prepare_data/<feature_set>/):
  - motif_enrichment.csv, count11.csv (loop x group), n_cells.tsv (cells per group),
    y_<condition>.csv
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from core.log import configure_logging
from human_prepare_data.anchor_openness import loop_activity
from human_prepare_data.constants import CONDITIONS, GROUP_SEPARATOR, POOLED_GROUP
from prepare_data.compute_motif_enrichment import compute_group_enrichment

logger = logging.getLogger(__name__)


def align_cells(openness_df: pd.DataFrame, motifs_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict both matrices to their shared cells, in the same order."""
    shared_cells = openness_df.columns.intersection(motifs_df.columns)
    logger.info(f"Shared cells: {len(shared_cells)} (openness {openness_df.shape[1]}, motifs {motifs_df.shape[1]})")

    if len(shared_cells) == 0:
        raise ValueError("No shared cells between anchor-openness and chromVAR matrices.")

    return openness_df[shared_cells], motifs_df[shared_cells]


def load_cell_groups(partition_path: Path, group_column: str, cells: pd.Index) -> pd.Series:
    """Map every cell to its group in the partition; fail if a cell has no group."""
    partition = pd.read_csv(partition_path, sep="\t", usecols=["barcode", group_column], dtype=str)
    if partition["barcode"].duplicated().any():
        raise ValueError(f"{partition_path} lists some barcodes more than once; a partition needs one group per cell.")

    groups = partition.set_index("barcode")[group_column].reindex(cells)
    n_missing = int(groups.isna().sum())
    if n_missing:
        raise ValueError(f"{n_missing} analysis cells have no '{group_column}' group in {partition_path}.")

    logger.info(f"Cell partition '{group_column}': {groups.nunique()} groups, sizes {groups.value_counts().to_dict()}")
    return groups


def compute_grouped_enrichment(activity_df: pd.DataFrame, motifs_df: pd.DataFrame,
                               groups: pd.Series | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Motif enrichment computed within each cell group, stacked side by side as features.

    Args:
        activity_df: loop x cell activity codes.
        motifs_df: motif x cell chromVAR scores, same cells as activity_df.
        groups: cell -> group, or None for one pooled group with plain motif feature names.

    Returns:
        enrichment (loop x feature) and count11 (loop x group: number of "1-1" cells).
    """
    pooled = groups is None
    if pooled:
        groups = pd.Series(POOLED_GROUP, index=activity_df.columns)

    enrichment_blocks, count11_by_group = [], {}
    for group in sorted(groups.unique()):
        in_group = (groups == group).to_numpy()
        enrichment_df, count_11 = compute_group_enrichment(activity_df.loc[:, in_group],
                                                           motifs_df.loc[:, in_group])
        if not pooled:
            enrichment_df.columns = [f"{motif}{GROUP_SEPARATOR}{group}" for motif in enrichment_df.columns]

        enrichment_blocks.append(enrichment_df)
        count11_by_group[group] = count_11
        logger.info(f"  group {group}: {int(in_group.sum())} cells, "
                    f"loops with >=1 both-open cell: {int((count_11 > 0).sum())} / {len(count_11)}")

    enrichment = pd.concat(enrichment_blocks, axis=1)
    count11 = pd.DataFrame(count11_by_group, index=activity_df.index)
    return enrichment, count11


def write_label_vectors(loop_table: pd.DataFrame, loop_ids: pd.Index, output_dir: Path) -> None:
    """Write y_<condition>.csv: each condition's positives (1) and clean negatives (0)."""
    for cond in CONDITIONS:
        # Each condition has its own loop set: positives (label 1) + clean
        # negatives (label 0). Loops excluded for this condition are NaN, so
        # dropping them makes y_<cond> restrict the shared enrichment matrix to
        # exactly this condition's loops (its own feature matrix, by row).
        y = loop_table.loc[loop_ids, f"label_{cond}"].dropna().astype(int)
        y.to_csv(output_dir / f"y_{cond}.csv")
        logger.info(f"  y_{cond}: {int(y.sum())} positives / {int((y == 0).sum())} negatives "
                    f"({len(y)} loops)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Human Stage 0: motif enrichment + loop presence vectors.")
    parser.add_argument("--anchor_openness", type=Path, default=Path("data/human_prototype/interim/anchor_openness.tsv"))
    parser.add_argument("--motifs", type=Path, default=Path("data/human_prototype/interim/motifs_chromvar.tsv"))
    parser.add_argument("--loop_table", type=Path, default=Path("data/human_prototype/interim/other_conditions/loop_universe.tsv"))
    parser.add_argument("--partition", type=Path, default=None,
                        help="Cell partition table (TSV with a barcode column); omit for pooled motif features.")
    parser.add_argument("--group_column", default=None, help="Column of --partition holding each cell's group.")
    parser.add_argument("--output_dir", type=Path, default=Path("results/human_prototype/other_conditions/prepare_data/motif"))
    parser.add_argument("--log_path", type=Path, default=Path("logs/human_prototype/stage0.log"))
    args = parser.parse_args()

    if (args.partition is None) != (args.group_column is None):
        parser.error("--partition and --group_column must be given together")

    configure_logging(args.log_path)
    openness_df = pd.read_csv(args.anchor_openness, sep="\t", index_col=[0, 1, 2]).astype("int8")
    motifs_df = pd.read_csv(args.motifs, sep="\t", index_col=0)
    loop_table = pd.read_csv(args.loop_table, sep="\t")

    openness_df, motifs_df = align_cells(openness_df, motifs_df)
    activity_df = loop_activity(loop_table, openness_df)

    groups = (load_cell_groups(args.partition, args.group_column, activity_df.columns)
              if args.partition is not None else None)

    logger.info("Computing motif enrichment (reusing compute_group_enrichment) ...")
    enrichment_df, count11 = compute_grouped_enrichment(activity_df, motifs_df, groups)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    enrichment_df.to_csv(args.output_dir / "motif_enrichment.csv")
    logger.info(f"Wrote motif_enrichment.csv  shape={enrichment_df.shape}")
    count11.to_csv(args.output_dir / "count11.csv")

    # Record how many cells each group has, for Stage 1 titles.
    n_cells = (groups.value_counts() if groups is not None
               else pd.Series({POOLED_GROUP: activity_df.shape[1]}))
    n_cells.sort_index().rename_axis("group").rename("n_cells").to_csv(args.output_dir / "n_cells.tsv", sep="\t")

    write_label_vectors(loop_table.set_index("loop_id"), enrichment_df.index, args.output_dir)
    logger.info("Stage 0 done.")


if __name__ == "__main__":
    main()
