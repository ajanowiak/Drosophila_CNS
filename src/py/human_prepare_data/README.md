# human_prepare_data

Human-brain prototype: the upstream steps that turn a GSE244618 `.snap` and the
Rahman et al. 2023 loop calls into the two matrices the Drosophila Stage 0/1
consume, plus thin drivers that reuse those stages unchanged.

The Drosophila pipeline is never modified: `run_stage0_human.py` reuses
`prepare_data.compute_motif_enrichment.compute_group_enrichment` and
`run_stage1_human.py` reuses `train_full_models.cv` / `.plotting`.

## Running

The pipeline is a Snakemake workflow (`src/snakemake/human_prototype/`), run
from the repo root:

    bash run_human_snakemake.sh 4        # 4 cores
    bash run_human_snakemake.sh 4 -n     # dry run

It produces nine classifiers: 3 ATAC cell sets {neuron, non_neuron, all} x 3
loop-presence labels {fetal, glia, neuron}. chromVAR is computed once on all QC
nuclei (one shared motif x cell feature set); the cell sets differ only in which
cells define anchor openness. Parameters, sample and paths live in
`config/human_prototype.yml`; every step runs in the `human_prep` env
(`env/human_prep.yaml`).

## Scripts

- `constants.py` - shared constants (conditions, cell sets, atlas classes, main
  chromosomes, plot display names).
- `build_loop_universe.py` - unified loop universe with per-condition presence,
  from the three Rahman loop BEDs (reciprocal anchor-overlap clustering).
- `snap_io.py` - readers for the `.snap` HDF5 (barcodes, cell x peak + peaks,
  cell x gene, per-cell fragments).
- `annotate_cells.py` - QC filter + per-nucleus labels joined from the Li 2023
  atlas metatable: `cellclass`, `subclass` (42), `celltype` (107), plus a coarse
  neuron / non_neuron `label`.
- `compute_chromvar.py` - chromVAR cell x motif deviations (hg38, JASPAR) via
  pychromvar.
- `anchor_openness.py` - loop x cell anchor-activity matrix (11/10/1/0) from
  fragments.
- `run_stage0_human.py` - Stage 0 driver: motif-enrichment matrix + per-condition
  loop-presence vectors.
- `run_stage1_human.py` - Stage 1 driver: condition-specific XGBoost + ROC plots.
- `aggregate_results.py` - collate the nine (cell set x label) classifiers into
  one ROC-AUC grid.
- `inspect_snap.py` - utility (not part of the pipeline): dump a `.snap` file's
  HDF5 structure.
- `survey_pfc_snaps.py` - one-off helper (not part of the pipeline) used to pick
  the sample from the PFC-neocortex libraries.

## TODO

- Move the generic `.snap` HDF5 readers in `snap_io.py` into a shared `io`
  module (alongside `prepare_data/io.py`) instead of keeping them local to this
  prototype.
