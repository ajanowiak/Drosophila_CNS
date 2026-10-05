The `env/` directory contains conda environments.

- `prepare_data.yaml` - Stage 0 (motif enrichment, loop presence vectors).
  Needs `pyreadr` to read the RDS metadata file.
- `analysis.yaml` - Stages 1-4 (model training, SHAP, first/second logit
  model). All four stages fit models and/or plot results, so they share one
  environment.
- `human_prep.yaml` - the human-brain prototype (`src/snakemake/human_prototype`):
  reading the `.snap` HDF5, loop-anchor overlap (`pybedtools` for the loop universe,
  `pyranges` for anchor openness), and the reused Stage 0/1 math.
- `human_prep_r.yaml` - the chromVAR step (`src/R/human_prepare_data/compute_chromvar.R`):
  Bioconductor `chromVAR` / `motifmatchr` / `TFBSTools`, with `rhdf5` to read the
  `.snap` and `Rsamtools` to read hg38.

