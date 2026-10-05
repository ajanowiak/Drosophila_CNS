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
loop labels {fetal, glia, neuron}. Each condition has its OWN loop set -
positives are that condition's loops (native coordinates) and negatives are the
other conditions' loops whose Hi-C rectangle coincides with NO actual loop of this
condition (the two anchors are never paired together here; see
`build_loop_universe.py`). chromVAR is computed once on all QC nuclei
(one shared motif x cell feature set); the cell sets differ only in which cells
define anchor openness. Parameters, sample and paths live in
`config/human_prototype.yml`. Most steps run in the `human_prep` env
(`env/human_prep.yaml`); the chromVAR step is in R (`src/R/human_prepare_data/`,
`env/human_prep_r.yaml`).

## Scripts

- `constants.py` - shared constants (conditions, cell sets, atlas classes, main
  chromosomes, plot display names).
- `download.py` - download one input file: this sample's `.snap.gz` from GEO
  (config `snap_url`, with `--gunzip`), a Rahman loop BED from the public S3
  bucket (config `loops_base_url`, kept gzipped), the JASPAR PFMs (`jaspar_url`),
  or the hg38 FASTA (`genome_url`, with `--gunzip --faidx` so the `.fai` chromVAR
  needs is written too). Each has a `download_*` Snakemake rule whose output is the
  exact config path the pipeline reads, so a fresh host fetches these automatically.
  The CATlas atlas metatable (`annotation`) has no stable URL and is placed on the
  host manually.
- `build_loop_universe.py` - per-condition loop sets (positives + disjoint-
  rectangle negatives) from the three Rahman loop BEDs; one table, one label
  column per condition. Each loop is a rectangle on the Hi-C plane (anchor 1 x
  anchor 2); a cross-condition loop is EXCLUDED when its rectangle overlaps a real
  loop of the condition (both anchors paired as in that condition), else it is a
  NEGATIVE. The every-loop-vs-every-loop 2-D overlap test is a BEDPE `pairToPair
  -type both` join (bedtools indexes + sweeps, output-sensitive), not an O(N*M)
  product. Parameter-free; anchors are highly conserved across conditions, so the
  signal is in the pairing, not the anchors.
- `snap_io.py` - readers for the `.snap` HDF5 (barcodes, cell x peak + peaks,
  cell x gene, per-cell fragments).
- `annotate_cells.py` - QC filter + per-nucleus labels joined from the Li 2023
  atlas metatable: `cellclass`, `subclass` (42), `celltype` (107), plus a coarse
  neuron / non_neuron `label`.
- `anchor_openness.py` - loop x cell anchor-activity matrix (11/10/1/0) from
  fragments.
- `run_stage0_human.py` - Stage 0 driver: motif-enrichment matrix + per-condition
  label vectors (each condition's positives + clean negatives).
- `run_stage1_human.py` - Stage 1 driver: condition-specific XGBoost + ROC plots.
- `aggregate_results.py` - collate the nine (cell set x label) classifiers into
  one ROC-AUC grid.
- `inspect_snap.py` - utility (not part of the pipeline): dump a `.snap` file's
  HDF5 structure.
- `survey_pfc_snaps.py` - one-off helper (not part of the pipeline) used to pick
  the sample from the PFC-neocortex libraries.

The chromVAR step is in R: `src/R/human_prepare_data/compute_chromvar.R` reads the
`.snap` peak matrix and writes `motifs_chromvar_all.tsv` (see `src/R/README.md`).

## TODO

- Move the generic `.snap` HDF5 readers in `snap_io.py` into a shared `io`
  module (alongside `prepare_data/io.py`) instead of keeping them local to this
  prototype.
