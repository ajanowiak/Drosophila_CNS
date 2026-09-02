The `env/` directory contains conda environments.

- `prepare_data.yaml` - Stage 0 (motif enrichment, loop presence vectors).
  Needs `pyreadr` to read the RDS metadata file.
- `analysis.yaml` - Stages 1-4 (model training, SHAP, first/second logit
  model). All four stages fit models and/or plot results, so they share one
  environment.

