#!/bin/bash

snakemake \
    --snakefile src/snakemake/Snakefile \
    --cores 1 \
    --printshellcmds \
    --reason