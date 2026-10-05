# src/R

R code for the human-brain prototype: the chromVAR step of the pipeline.

## human_prepare_data/compute_chromvar.R

Computes the motif x cell chromVAR deviation matrix (Schep et al. 2017) with the
Bioconductor `chromVAR` package. It reads the SnapATAC `.snap` peak matrix, keeps
the selected cells and accessible main-chromosome peaks, scores the JASPAR CORE
vertebrate motifs (a PFM flat file read with `TFBSTools::readJASPARMatrix`), and
writes `motifs_chromvar_all.tsv` (motif x cell), which Stage 0 consumes.

Wired through Snakemake (the `compute_chromvar` rule in
`src/snakemake/human_prototype/prepare_data.smk`; `download_jaspar` fetches the
motif file); environment is `env/human_prep_r.yaml`. To run it alone:

    Rscript src/R/human_prepare_data/compute_chromvar.R \
        --snap data/human_prototype/raw/GSM7822145_MM_417.snap \
        --metadata data/human_prototype/interim/cell_metadata.tsv \
        --genome data/human_prototype/ref/hg38.fa \
        --motifs data/human_prototype/ref/JASPAR2026_CORE_vertebrates.jaspar \
        --cell_label all \
        --output data/human_prototype/interim/motifs_chromvar_all.tsv
