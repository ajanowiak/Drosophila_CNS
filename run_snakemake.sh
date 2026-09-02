#!/bin/bash

# Usage: run_snakemake.sh [cores] [extra snakemake args...]
# Defaults to 1 core; pass the number of cores to use as the first argument,
# e.g. `run_snakemake.sh 8`. Any further arguments are forwarded to
# snakemake as-is (e.g. `run_snakemake.sh 8 -n` for a dry run).

cores="${1:-1}"
shift || true

snakemake \
    --snakefile src/snakemake/Snakefile \
    --use-conda \
    --cores "$cores" \
    --printshellcmds \
    --keep-going \
    "$@" \
    all