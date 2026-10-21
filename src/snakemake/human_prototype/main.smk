configfile: "config/human_prototype.yml"

NEGATIVE_MODES = ["other_conditions", "n2"]   # human_prepare_data.constants.NEGATIVE_MODES

# `--config negatives=n2` on the CLI gives a single string; treat it as a one-mode list.
if isinstance(config["negatives"], str):
    config["negatives"] = [config["negatives"]]
unknown_modes = [mode for mode in config["negatives"] if mode not in NEGATIVE_MODES]
if unknown_modes:
    raise ValueError(f"config negatives must be among {NEGATIVE_MODES}, got {unknown_modes!r}")

FEATURE_SETS = list(config["feature_sets"])

# {negatives} namespaces every output from the loop universe on (see config);
# {features} names the Stage 0 feature set (see config feature_sets).
wildcard_constraints:
    negatives="|".join(NEGATIVE_MODES),
    features="|".join(FEATURE_SETS),


# One target-list function per stage, defined here (before any include) so both
# the master `rule all` below and each stage's own `all_<stage>` rule (in its
# .smk) share a single definition instead of two copies that could drift apart.

def prepare_data_targets():
    return expand(
        "results/human_prototype/{negatives}/prepare_data/{features}/motif_enrichment.csv",
        negatives=config["negatives"],
        features=FEATURE_SETS,
    )


def train_targets():
    return expand(
        "results/human_prototype/{negatives}/train/{features}/figures/roc_combined.png",
        negatives=config["negatives"],
        features=FEATURE_SETS,
    ) + ["results/human_prototype/train_summary.csv"]


def logit_targets():
    return expand(
        "results/human_prototype/{negatives}/logit/{features}/summary_{condition}.csv",
        negatives=config["negatives"],
        features=config["logit_feature_sets"],
        condition=config["conditions"],
    )


# rule all must stay the first rule in the workflow (across all includes) so it
# remains Snakemake's default target when none is given on the CLI.
rule all:
    input:
        prepare_data_targets() + train_targets() + logit_targets()


# One file per stage, mirroring the src/py/human_prepare_data layout.
include: "prepare_data.smk"
include: "train.smk"
include: "logit.smk"
