#!/bin/bash

snakemake \
    --snakefile src/snakemake/Snakefile \
    --use-conda \
    --cores 1 \
    --printshellcmds \
    --reason \
    --keep-going \
    all