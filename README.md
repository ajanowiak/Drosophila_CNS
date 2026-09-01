# Drosophila_CNS

Prediction of chromatin loop formation in the central nervous system cells
of *Drosophila melanogaster* based on scATAC-seq data.

## Running the pipeline

```bash
bash run_snakemake.sh
```

Runs every stage end-to-end via Snakemake, using conda envs per stage. Set
`n_splits`, model list, tissues, etc. in `config/config.yml`.

To run one stage only:

```bash
snakemake -snakefile src/snakemake/Snakefile -use-conda -cores 1 all_prepare_data
```

Replace `all_prepare_data` with `all_train_full_models`,
`all_shap_importance`, `all_first_logit_model`, or `all_second_logit_model`.

## Pipeline stages

0. **prepare_data** - motif enrichment matrices, loop presence labels.
1. **train_full_models** - time-specific and time-agnostic classifiers.
2. **shap_importance** - SHAP importance for the time-agnostic classifiers.
3. **first_logit_model** - SHAP-ranked feature preselection, Logit fit.
4. **second_logit_model** - refit on the BH-significant motifs.

See [src/py/README.md](src/py/README.md) for what each script does.

## Repo layout

- `config/` - `config.yml`, the pipeline's single config file.
- `data/` - input data (not tracked; see `data/.gitignore`).
- `env/` - conda environments (`prepare_data.yaml`, `analysis.yaml`).
- `results/` - pipeline output.
- `src/py/` - pipeline code.
- `src/snakemake/` - Snakemake rules, one subdirectory per stage.

- `doc/` - documentation.

