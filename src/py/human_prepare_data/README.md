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

It trains one classifier per loop label {fetal, glia, neuron} for every
(negative mode x feature set) pair listed in the config. Each condition has its OWN loop set -
positives are that condition's loops (native coordinates); the negatives depend
on the config key `negatives` (see `build_loop_universe.py`):

- `other_conditions` - the other conditions' loops whose Hi-C rectangle coincides
  with NO actual loop of this condition (the two anchors are never paired
  together here);
- `n2` - `new_pairs_of_anchors`: random new pairs of this condition's
  own anchors, `neg_pos_ratio[condition]` per positive, drawn with `random_state`.
  The per-condition ratios are set to `other_conditions`' negatives / positives,
  so both modes have the same class balance.

Everything from the loop universe on is written under
`data/human_prototype/interim/<negatives>/` and `results/human_prototype/<negatives>/`
(logs under `logs/human_prototype/<negatives>/`), so the two modes coexist. Both
listed modes run by default; run one with e.g. `bash run_human_snakemake.sh 4 --config negatives=n2`.

The feature sets (config `feature_sets`) differ in the cells enrichment is
computed within: `motif` contrasts the both-open "1-1" cells with all other
analysis cells (one feature per motif); a partitioned set such as
`motif_x_cellclass` computes that contrast within each group of a cell
partition (here the atlas `cellclass`: GLUT / GABA / NonN) and stacks the
groups (one feature per motif x group, named `<motif>__<group>`). The partition
is an input (a barcode -> group table and column), so a coarser grouping of the
atlas cell types can replace it without code changes.

The analysis cells are the QC-passing, atlas-annotated nuclei (`in_analysis`
in `cell_metadata.tsv`; the unannotated ones are dropped and counted in the
annotate_cells log). chromVAR is computed once on them (one shared motif x cell
matrix), and anchor openness once per distinct anchor, shared by both negative
modes. Parameters, sample and paths live in
`config/human_prototype.yml`. Most steps run in the `human_prep` env
(`env/human_prep.yaml`); the chromVAR step is in R (`src/R/human_prepare_data/`,
`env/human_prep_r.yaml`).

## Scripts

- `constants.py` - shared constants (conditions, negative modes, atlas columns,
  main chromosomes, feature naming).
- `download.py` - download one input file: this sample's `.snap.gz` from GEO
  (config `snap_url`, with `--gunzip`), a Rahman loop BED from the public S3
  bucket (config `loops_base_url`, kept gzipped), the JASPAR PFMs (`jaspar_url`),
  or the hg38 FASTA (`genome_url`, with `--gunzip --faidx` so the `.fai` chromVAR
  needs is written too). Each has a `download_*` Snakemake rule whose output is the
  exact config path the pipeline reads, so a fresh host fetches these automatically.
  The CATlas atlas metatable (`annotation`) has no stable URL and is placed on the
  host manually.
- `build_loop_universe.py` - per-condition loop sets (positives + negatives) from
  the three Rahman loop BEDs; one table, one label column per condition. Each
  loop is a rectangle on the Hi-C plane (anchor 1 x anchor 2). Mode
  `other_conditions`: a cross-condition loop is EXCLUDED when its rectangle
  overlaps a real loop of the condition (both anchors paired as in that
  condition), else it is a NEGATIVE. The every-loop-vs-every-loop 2-D overlap
  test is a BEDPE `pairToPair -type both` join (bedtools indexes + sweeps,
  output-sensitive), not an O(N*M) product. Parameter-free; anchors are highly
  conserved across conditions, so the signal is in the pairing, not the anchors.
  Mode `n2`: every non-overlapping same-chromosome pair of the condition's own
  anchors is a candidate; real loops are removed, `int(neg_pos_ratio * #loops)`
  are drawn uniformly (seeded), and drawn pairs whose rectangle overlaps a real
  loop are dropped (same `pairToPair` test) - in the same order as the [original repo](https://github.com/ola-cupriak/predicting-the-formation-of-chromatin-loops-using-genomic-data). Each n2
  negative is labelled for its own condition only (`source` = `n2_<condition>`).
- `snap_io.py` - readers for the `.snap` HDF5 (barcodes, cell x peak + peaks,
  cell x gene, per-cell fragments).
- `annotate_cells.py` - QC filter + per-nucleus labels joined from the Li 2023
  atlas metatable: `cellclass`, `subclass` (42), `celltype` (107), plus the
  `annotated` and `in_analysis` (QC-passing and annotated) flags.
- `anchor_openness.py` - anchor x cell openness matrix (0/1) from fragments, once
  per distinct anchor of the three conditions' loops; `loop_activity` turns it
  into a loop universe's loop x cell activity codes (11/10/1/0).
- `run_stage0_human.py` - Stage 0 driver: one feature set's motif-enrichment
  matrix (pooled, or per cell group) + per-condition label vectors (each
  condition's positives + clean negatives).
- `run_stage1_human.py` - Stage 1 driver: one condition's XGBoost + ROC plot
  (Snakemake runs one job per negative mode x feature set x condition).
- `summarize_stage1.py` - collect one feature set's conditions into
  `cv_summary.csv` and the combined ROC figure.
- `aggregate_results.py` - collate every (negative mode x feature set x label)
  classifier into one table and ROC-AUC grid.
- `inspect_snap.py` - utility (not part of the pipeline): dump a `.snap` file's
  HDF5 structure.
- `survey_pfc_snaps.py` - one-off helper (not part of the pipeline) used to pick
  the sample from the PFC-neocortex libraries.

The chromVAR step is in R: `src/R/human_prepare_data/compute_chromvar.R` reads the
`.snap` peak matrix and writes `motifs_chromvar.tsv` (see `src/R/README.md`).

## TODO

- Move the generic `.snap` HDF5 readers in `snap_io.py` into a shared `io`
  module (alongside `prepare_data/io.py`) instead of keeping them local to this
  prototype.
