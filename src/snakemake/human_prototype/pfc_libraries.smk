# Prototype, standalone (not included in the main human Snakefile): per-library
# Stage 0 for the GSE244618 PFC libraries in config/human_pfc_libraries.yml.
# The main pipeline's steps run separately inside each library - QC + atlas labels,
# chromVAR, anchor openness and the motif-enrichment matrices are computed within one
# library's cells - against the main pipeline's loop universe, so every library
# scores the same loops. Two feature sets per library: pooled motifs, and motif x
# cell class (enrichment within each class of the library's own cells).
#
# The shared inputs (hg38 + .fai, JASPAR, the Rahman loop BEDs, the atlas metatable
# and the loop universe) are not built here; they must already exist from the main
# pipeline. Run from the repo root:
#
#   snakemake --snakefile src/snakemake/human_prototype/pfc_libraries.smk --use-conda --cores 16

import os

configfile: "config/human_prototype.yml"
configfile: "config/human_pfc_libraries.yml"

PFC = config["pfc"]
LIBRARIES = list(PFC["libraries"])

if config["feature_sets"][PFC["feature_set"]]:
    raise ValueError(f"pfc.feature_set must be a feature set without a cell partition, got {PFC['feature_set']!r}")

CELLCLASS_SET = config["feature_sets"][PFC["cellclass_feature_set"]]
if not CELLCLASS_SET:
    raise ValueError(f"pfc.cellclass_feature_set must be a feature set with a cell partition, "
                     f"got {PFC['cellclass_feature_set']!r}")

LIBRARY_INTERIM = "data/human_prototype/interim/libraries/{library}"
LIBRARY_STAGE0 = f"results/human_prototype/{PFC['negatives']}/libraries/{{library}}/prepare_data/{PFC['feature_set']}"
LIBRARY_CELLCLASS_STAGE0 = (f"results/human_prototype/{PFC['negatives']}/libraries/{{library}}/prepare_data/"
                            f"{PFC['cellclass_feature_set']}")

wildcard_constraints:
    library="|".join(LIBRARIES),


def library_snap(wildcards):
    """The library's decompressed .snap path (raw_dir/<GSM>_<library>.snap)."""
    gsm = PFC["libraries"][wildcards.library]["gsm"]
    return f"{config['raw_dir']}/{gsm}_{wildcards.library}.snap"


def library_snap_url(wildcards):
    """GEO supplementary URL of the library's .snap.gz; checks the GSM matches the config."""
    gsm = PFC["libraries"][wildcards.library]["gsm"]
    if gsm != wildcards.gsm:
        raise ValueError(f"{wildcards.library} is {gsm} in config pfc.libraries, not {wildcards.gsm}")
    return PFC["snap_url_template"].format(gsm_stub=f"{gsm[:-3]}nnn", gsm=gsm, library=wildcards.library)


rule pfc:
    input:
        expand(f"{LIBRARY_STAGE0}/motif_enrichment.csv", library=LIBRARIES),
        expand(f"{LIBRARY_CELLCLASS_STAGE0}/motif_enrichment.csv", library=LIBRARIES),
        expand(f"{LIBRARY_INTERIM}/anchor_openness_summary.tsv", library=LIBRARIES),


# Download + decompress one library's .snap from GEO.
rule download_library_snap:
    output:
        f"{config['raw_dir']}/{{gsm}}_{{library}}.snap"
    params:
        url=library_snap_url,
    wildcard_constraints:
        gsm=r"GSM\d+",
    log:
        "logs/human_prototype/libraries/{library}/download_snap_{gsm}.log"
    conda:
        "../../../env/human_prep.yaml"
    shell:
        """
        PYTHONPATH=src/py python src/py/human_prepare_data/download.py \
            --url {params.url} \
            --output {output} \
            --gunzip \
            --log_path {log}
        """


# Per-nucleus QC + Li 2023 atlas labels for one library.
rule annotate_library_cells:
    input:
        snap=library_snap,
        annotation=config["annotation"],
    output:
        f"{LIBRARY_INTERIM}/cell_metadata.tsv"
    log:
        "logs/human_prototype/libraries/{library}/annotate_cells.log"
    conda:
        "../../../env/human_prep.yaml"
    shell:
        """
        PYTHONPATH=src/py python src/py/human_prepare_data/annotate_cells.py \
            --snap {input.snap} \
            --sample {wildcards.library} \
            --annotation {input.annotation} \
            --min_fragments {config[min_fragments]} \
            --output {output} \
            --log_path {log}
        """


# chromVAR on one library's analysis nuclei, with that library's own peaks and
# background peak sets.
rule library_chromvar:
    input:
        snap=library_snap,
        genome=config["genome"],
        genome_fai=f"{config['genome']}.fai",
        motifs=config["jaspar"],
        metadata=f"{LIBRARY_INTERIM}/cell_metadata.tsv",
    output:
        f"{LIBRARY_INTERIM}/motifs_chromvar.tsv"
    threads: config["blas_threads"]
    log:
        "logs/human_prototype/libraries/{library}/compute_chromvar.log"
    conda:
        "../../../env/human_prep_r.yaml"
    shell:
        """
        export OPENBLAS_NUM_THREADS={threads} OMP_NUM_THREADS={threads} MKL_NUM_THREADS={threads}
        Rscript src/R/human_prepare_data/compute_chromvar.R \
            --snap {input.snap} \
            --metadata {input.metadata} \
            --genome {input.genome} \
            --motifs {input.motifs} \
            --min_cells_per_peak {config[min_cells_per_peak]} \
            --bg_iterations {config[bg_iterations]} \
            --output {output} \
            --log_path {log}
        """


# Anchor x cell openness for one library's analysis nuclei.
rule library_anchor_openness:
    input:
        snap=library_snap,
        loop_beds=expand(f"{config['raw_dir']}/{{condition}}_loop.bed.gz", condition=config["conditions"]),
        metadata=f"{LIBRARY_INTERIM}/cell_metadata.tsv",
    output:
        f"{LIBRARY_INTERIM}/anchor_openness.tsv"
    log:
        "logs/human_prototype/libraries/{library}/anchor_openness.log"
    conda:
        "../../../env/human_prep.yaml"
    shell:
        """
        PYTHONPATH=src/py python src/py/human_prepare_data/anchor_openness.py \
            --snap {input.snap} \
            --loop_beds {input.loop_beds} \
            --metadata {input.metadata} \
            --output {output} \
            --log_path {log}
        """


# Open-cell counts per anchor (all cells and per cell class) for one library: a small
# table for the anchor-openness notebook, which cannot reload the 2.4 GB matrix quickly.
rule library_anchor_openness_summary:
    input:
        openness=f"{LIBRARY_INTERIM}/anchor_openness.tsv",
        metadata=f"{LIBRARY_INTERIM}/cell_metadata.tsv",
    output:
        f"{LIBRARY_INTERIM}/anchor_openness_summary.tsv"
    params:
        group_column=CELLCLASS_SET["column"],
    log:
        "logs/human_prototype/libraries/{library}/anchor_openness_summary.log"
    conda:
        "../../../env/human_prep.yaml"
    shell:
        """
        PYTHONPATH=src/py python src/py/human_prepare_data/summarize_anchor_openness.py \
            --anchor_openness {input.openness} \
            --metadata {input.metadata} \
            --group_column {params.group_column} \
            --output {output} \
            --log_path {log}
        """


# Motif enrichment + per-condition loop-presence vectors for one library.
rule library_stage0:
    input:
        openness=f"{LIBRARY_INTERIM}/anchor_openness.tsv",
        motifs=f"{LIBRARY_INTERIM}/motifs_chromvar.tsv",
        loop_table=f"data/human_prototype/interim/{PFC['negatives']}/loop_universe.tsv",
    output:
        enrichment=f"{LIBRARY_STAGE0}/motif_enrichment.csv",
        count11=f"{LIBRARY_STAGE0}/count11.csv",
        n_cells=f"{LIBRARY_STAGE0}/n_cells.tsv",
        y=[f"{LIBRARY_STAGE0}/y_{cond}.csv" for cond in config["conditions"]],
    params:
        output_dir=lambda wildcards, output: os.path.dirname(output.enrichment),
    threads: config["stage0_threads"]
    log:
        f"logs/human_prototype/{PFC['negatives']}/libraries/{{library}}/stage0/{PFC['feature_set']}.log"
    conda:
        "../../../env/human_prep.yaml"
    shell:
        """
        export OPENBLAS_NUM_THREADS={threads} OMP_NUM_THREADS={threads} MKL_NUM_THREADS={threads}
        PYTHONPATH=src/py python src/py/human_prepare_data/run_stage0_human.py \
            --anchor_openness {input.openness} \
            --motifs {input.motifs} \
            --loop_table {input.loop_table} \
            --output_dir {params.output_dir} \
            --log_path {log}
        """


# Motif x cell-class enrichment for one library: "1-1" vs other within each cell
# class of the library's own analysis cells, one feature per motif x class. Kept
# apart from library_stage0 so the pooled outputs are not rerun.
rule library_stage0_cellclass:
    input:
        openness=f"{LIBRARY_INTERIM}/anchor_openness.tsv",
        motifs=f"{LIBRARY_INTERIM}/motifs_chromvar.tsv",
        loop_table=f"data/human_prototype/interim/{PFC['negatives']}/loop_universe.tsv",
        partition=f"{LIBRARY_INTERIM}/cell_metadata.tsv",
    output:
        enrichment=f"{LIBRARY_CELLCLASS_STAGE0}/motif_enrichment.csv",
        count11=f"{LIBRARY_CELLCLASS_STAGE0}/count11.csv",
        n_cells=f"{LIBRARY_CELLCLASS_STAGE0}/n_cells.tsv",
        y=[f"{LIBRARY_CELLCLASS_STAGE0}/y_{cond}.csv" for cond in config["conditions"]],
    params:
        output_dir=lambda wildcards, output: os.path.dirname(output.enrichment),
        group_column=CELLCLASS_SET["column"],
    threads: config["stage0_threads"]
    log:
        f"logs/human_prototype/{PFC['negatives']}/libraries/{{library}}/stage0/{PFC['cellclass_feature_set']}.log"
    conda:
        "../../../env/human_prep.yaml"
    shell:
        """
        export OPENBLAS_NUM_THREADS={threads} OMP_NUM_THREADS={threads} MKL_NUM_THREADS={threads}
        PYTHONPATH=src/py python src/py/human_prepare_data/run_stage0_human.py \
            --anchor_openness {input.openness} \
            --motifs {input.motifs} \
            --loop_table {input.loop_table} \
            --partition {input.partition} \
            --group_column {params.group_column} \
            --output_dir {params.output_dir} \
            --log_path {log}
        """
