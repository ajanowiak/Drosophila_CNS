configfile: "config/human_prototype.yml"


# One target-list function per stage, defined here (before any include) so both
# the master `rule all` below and each stage's own `all_<stage>` rule (in its
# .smk) share a single definition instead of two copies that could drift apart.

def prepare_data_targets():
    return expand(
        "results/human_prototype/prepare_data/{cell_set}/motif_enrichment.csv",
        cell_set=config["cell_sets"],
    )


def train_targets():
    return expand(
        "results/human_prototype/train/{cell_set}/figures/roc_combined.png",
        cell_set=config["cell_sets"],
    ) + ["results/human_prototype/train/all_cellsets_summary.csv"]


# rule all must stay the first rule in the workflow (across all includes) so it
# remains Snakemake's default target when none is given on the CLI.
rule all:
    input:
        prepare_data_targets() + train_targets()


# One file per stage, mirroring the src/py/human_prepare_data layout.
include: "prepare_data.smk"
include: "train.smk"
