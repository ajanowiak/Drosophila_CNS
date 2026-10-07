#!/bin/bash

# Usage: run_human_snakemake.sh [cores] [extra snakemake args...]
# Defaults to 1 core; pass the number of cores to use as the first argument,
# e.g. `run_human_snakemake.sh 4`. Any further arguments are forwarded to
# snakemake as-is (e.g. `run_human_snakemake.sh 4 -n` for a dry run, or
# `run_human_snakemake.sh 4 --config negatives=n2`).

cores="${1:-1}"
shift || true

# The target comes before the forwarded args: --config consumes every word after it.
snakemake \
    all \
    --snakefile src/snakemake/human_prototype/Snakefile \
    --use-conda \
    --cores "$cores" \
    --printshellcmds \
    --keep-going \
    "$@"
