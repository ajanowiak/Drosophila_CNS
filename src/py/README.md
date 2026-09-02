# src/py

Python code for the pipeline. Run with `PYTHONPATH=src/py`. Organized by
stage; each stage is a directory with a CLI script per Snakemake rule.

Stage order: `prepare_data` -> `train_full_models` -> `shap_importance` ->
`first_logit_model` -> `second_logit_model`.

## core/

Shared code, importable from any stage.

- `constants.py` - windows, tissues, model registry, enums
  (`FilteringMode`, `FeatureMode`, `FeatureSelectionMode`).
- `config.py` - loads `config/config.yml`.
- `log.py` - logging setup (file + stdout).
- `features.py` - `stack_windows()`: builds the time-agnostic feature
  matrix (curr/prev/expanded) used by every stage after Stage 0.
- `motif_labels.py` - motif ID -> display name lookup.
- `logit_analysis.py` - statsmodels Logit fitting, CV, and the binary
  search feature-count selector. Shared by Stage 3 and Stage 4.
- `logit_plots.py` - coefficient and volcano plots for a fitted Logit
  summary.
- `paths.py` - builds every `results/` path. Single source of truth for the
  `results/<stage>/<rule>/...` layout, so it isn't re-derived independently
  in each stage's `io.py`.

## prepare_data/ - Stage 0

Builds the training data every later stage reads.

- `compute_motif_enrichment.py` - per-loop motif enrichment (mean chromVAR
  score of "1-1" cells minus all other cells), one matrix per window.
- `extract_loop_presence_vectors.py` - binary loop-presence labels per
  tissue and window, from the loop annotation TSV.
- `io.py` - shared load/save helpers for both scripts above.

## train_full_models/ - Stage 1

Trains classifiers and produces comparison plots.

- `train_time_specific.py` - one classifier per (model, tissue, window),
  trained on that window's own data.
- `train_time_agnostic.py` - one classifier per (model, tissue,
  feature_mode), trained on all three windows stacked together.
  feature_mode is curr / prev / expanded.
- `cv.py` - shared CV loop and ROC aggregation for both training scripts.
- `plotting.py` - ROC curve plot.
- `combined_roc_plot.py` - overlays the three tissues' time-specific ROC
  curves for one (model, window).
- `compare_bar_plots.py` - bar charts: time-specific vs. time-agnostic AUC,
  and curr vs. prev vs. expanded AUC.
- `io.py` - model/result persistence and summary loaders for the plots.

## shap_importance/ - Stage 2

SHAP importance for the time-agnostic (curr) classifiers.

- `shap_analysis.py` - entry point: loads a trained model, computes SHAP
  values, saves a beeswarm plot and a per-motif importance table.
- `explain.py` - SHAP value computation (TreeExplainer for RF/XGB,
  KernelExplainer for LR/SVM) and summary-table aggregation.
- `plotting.py` - beeswarm plot.
- `io.py` - model loading, table saving.

Only RF and XGB (TreeExplainer) run by default. KernelExplainer models (LR, SVM)
are too slow at this scale.

## first_logit_model/ - Stage 3

Preselects SHAP-ranked motifs and fits a statsmodels Logit to get
coefficients and p-values.

- `regression_coefs.py` - preselects N motifs (by EPV budget or by binary
  search), fits the Logit, saves the coefficient table and coefficient/
  volcano plots.
- `logit_regression_aucroc.py` - cross-validates the already-chosen
  feature set(s); in EPV mode sweeps every configured EPV value for one
  tissue, in binsearch mode evaluates the single binsearch selection.
- `features.py` - EPV-based feature count, SHAP-rank downsampling,
  selection-tag naming.
- `io.py` - summary table save/read.

## second_logit_model/ - Stage 4

Retrains the Logit on only the BH-significant motifs from Stage 3.

- `logit_regression_significant_features.py` - entry point: reads Stage
  3's summary, keeps BH-significant motifs, refits, cross-validates, saves
  results and plots.
- `io.py` - significant-feature extraction, summary/CV-result save.

## obsolete/

Scripts used for exploratory analyses during pipeline development that did not make it to its final version.

