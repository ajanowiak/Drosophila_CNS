# Stage 0 (human) - turn the SNAP + Rahman loop calls into the two matrices the
# Drosophila enrichment math consumes (a shared chromVAR motif matrix and a
# per-cell-set loop x cell openness matrix), then the motif-enrichment matrix
# and per-condition loop-presence vectors.

rule all_prepare_data:
    input:
        prepare_data_targets()


# Download + decompress this sample's .snap from GEO (only the current sample).
# Produces config["snap"], the input every SNAP-reading rule below consumes, so
# on a fresh host Snakemake fetches it automatically; skipped if it already exists.
rule download_snap:
    output:
        config["snap"]
    params:
        url=config["snap_url"],
    log:
        "logs/human_prototype/download_snap.log"
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


# Download the JASPAR CORE vertebrate motif PFMs (plain text) from the JASPAR site.
rule download_jaspar:
    output:
        config["jaspar"]
    params:
        url=config["jaspar_url"],
    log:
        "logs/human_prototype/download_jaspar.log"
    conda:
        "../../../env/human_prep.yaml"
    shell:
        """
        PYTHONPATH=src/py python src/py/human_prepare_data/download.py \
            --url {params.url} \
            --output {output} \
            --log_path {log}
        """


# Download one condition's Rahman loop BED (kept gzipped) from the public S3 bucket.
rule download_loop_bed:
    output:
        f"{config['raw_dir']}/{{condition}}_loop.bed.gz"
    params:
        url=lambda wildcards: f"{config['loops_base_url']}/{wildcards.condition}_loop.bed.gz",
    wildcard_constraints:
        condition="|".join(config["conditions"]),
    log:
        "logs/human_prototype/download_loop_bed/{condition}.log"
    conda:
        "../../../env/human_prep.yaml"
    shell:
        """
        PYTHONPATH=src/py python src/py/human_prepare_data/download.py \
            --url {params.url} \
            --output {output} \
            --log_path {log}
        """


# Download + decompress the hg38 FASTA and build its .fai index. compute_chromvar
# (R, FaFile) reads config["genome"] and needs the .fai sitting next to it, so both
# are declared as outputs and fetched automatically on a fresh host.
rule download_genome:
    output:
        fasta=config["genome"],
        fai=f"{config['genome']}.fai",
    params:
        url=config["genome_url"],
    log:
        "logs/human_prototype/download_genome.log"
    conda:
        "../../../env/human_prep.yaml"
    shell:
        """
        PYTHONPATH=src/py python src/py/human_prepare_data/download.py \
            --url {params.url} \
            --output {output.fasta} \
            --gunzip \
            --faidx \
            --log_path {log}
        """


# NOTE: the CATlas (Li 2023) metatable (config["annotation"]) has no stable download
# URL and is provided manually on the server, so it has no rule here; annotate_cells
# consumes it as an existing input (Snakemake errors clearly if it is absent).


# Per-condition Rahman loop sets (positives + clean negatives), one table with a
# label column per condition; built once.
rule build_loop_universe:
    input:
        fetal=f"{config['raw_dir']}/fetal_loop.bed.gz",
        glia=f"{config['raw_dir']}/glia_loop.bed.gz",
        neuron=f"{config['raw_dir']}/neuron_loop.bed.gz",
    output:
        "data/human_prototype/interim/loop_universe.tsv"
    log:
        "logs/human_prototype/build_loop_universe.log"
    conda:
        "../../../env/human_prep.yaml"
    shell:
        """
        PYTHONPATH=src/py python src/py/human_prepare_data/build_loop_universe.py \
            --raw_dir {config[raw_dir]} \
            --output {output} \
            --log_path {log}
        """


# Per-nucleus QC + Li 2023 atlas labels; built once.
rule annotate_cells:
    input:
        snap=config["snap"],
        annotation=config["annotation"],
    output:
        "data/human_prototype/interim/cell_metadata.tsv"
    log:
        "logs/human_prototype/annotate_cells.log"
    conda:
        "../../../env/human_prep.yaml"
    shell:
        """
        PYTHONPATH=src/py python src/py/human_prepare_data/annotate_cells.py \
            --snap {input.snap} \
            --sample {config[sample]} \
            --annotation {input.annotation} \
            --min_fragments {config[min_fragments]} \
            --output {output} \
            --log_path {log}
        """


# chromVAR ONCE on all QC nuclei -> one shared motif x cell feature set. Every
# cell set reuses this matrix and just selects its own cells at Stage 0, so the
# same cell is never scored twice under different peak filters / backgrounds.
rule compute_chromvar:
    input:
        snap=config["snap"],
        genome=config["genome"],
        genome_fai=f"{config['genome']}.fai",
        motifs=config["jaspar"],
        metadata="data/human_prototype/interim/cell_metadata.tsv",
    output:
        "data/human_prototype/interim/motifs_chromvar_all.tsv"
    log:
        "logs/human_prototype/compute_chromvar.log"
    conda:
        "../../../env/human_prep_r.yaml"
    shell:
        """
        export OPENBLAS_NUM_THREADS={config[blas_threads]} OMP_NUM_THREADS={config[blas_threads]}
        Rscript src/R/human_prepare_data/compute_chromvar.R \
            --snap {input.snap} \
            --metadata {input.metadata} \
            --genome {input.genome} \
            --motifs {input.motifs} \
            --cell_label all \
            --min_cells_per_peak {config[min_cells_per_peak]} \
            --bg_iterations {config[bg_iterations]} \
            --output {output} \
            --log_path {log}
        """


# Loop x cell anchor openness, per ATAC cell set.
rule anchor_openness:
    input:
        snap=config["snap"],
        loops="data/human_prototype/interim/loop_universe.tsv",
        metadata="data/human_prototype/interim/cell_metadata.tsv",
    output:
        "data/human_prototype/interim/loops_activity_{cell_set}.tsv"
    log:
        "logs/human_prototype/anchor_openness/{cell_set}.log"
    conda:
        "../../../env/human_prep.yaml"
    shell:
        """
        PYTHONPATH=src/py python src/py/human_prepare_data/anchor_openness.py \
            --snap {input.snap} \
            --loops {input.loops} \
            --cell_label {wildcards.cell_set} \
            --metadata {input.metadata} \
            --output {output} \
            --log_path {log}
        """


# Motif enrichment + per-condition loop-presence vectors, per ATAC cell set.
rule run_stage0:
    input:
        loops="data/human_prototype/interim/loops_activity_{cell_set}.tsv",
        motifs="data/human_prototype/interim/motifs_chromvar_all.tsv",
        loop_table="data/human_prototype/interim/loop_universe.tsv",
    output:
        enrichment="results/human_prototype/prepare_data/{cell_set}/motif_enrichment.csv",
        count11="results/human_prototype/prepare_data/{cell_set}/count11.csv",
        n_cells="results/human_prototype/prepare_data/{cell_set}/n_cells.txt",
        y=expand("results/human_prototype/prepare_data/{{cell_set}}/y_{cond}.csv",
                 cond=config["conditions"]),
    log:
        "logs/human_prototype/stage0/{cell_set}.log"
    conda:
        "../../../env/human_prep.yaml"
    shell:
        """
        PYTHONPATH=src/py python src/py/human_prepare_data/run_stage0_human.py \
            --loops {input.loops} \
            --motifs {input.motifs} \
            --loop_table {input.loop_table} \
            --output_dir results/human_prototype/prepare_data/{wildcards.cell_set} \
            --log_path {log}
        """
