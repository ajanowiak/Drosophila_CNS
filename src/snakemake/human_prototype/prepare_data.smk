# Stage 0 (human) - turn the SNAP + Rahman loop calls into the two matrices the
# Drosophila enrichment math consumes (a shared chromVAR motif matrix and a
# shared anchor x cell openness matrix, from which each loop universe's loop x
# cell activity codes are built), then one motif-enrichment matrix per
# (negative mode x feature set) and per-condition loop-presence vectors.

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


# Per-condition Rahman loop sets (positives + negatives drawn by the {negatives}
# mode), one table with a label column per condition; built once per mode.
rule build_loop_universe:
    input:
        fetal=f"{config['raw_dir']}/fetal_loop.bed.gz",
        glia=f"{config['raw_dir']}/glia_loop.bed.gz",
        neuron=f"{config['raw_dir']}/neuron_loop.bed.gz",
    output:
        "data/human_prototype/interim/{negatives}/loop_universe.tsv"
    params:
        # Only n2 draws at random, so only n2 reruns when these change.
        n2_args=lambda wildcards: (
            "--neg_pos_ratio "
            + " ".join(f"{cond}={ratio}" for cond, ratio in config["neg_pos_ratio"].items())
            + f" --random_state {config['random_state']}"
            if wildcards.negatives == "n2" else ""
        ),
    log:
        "logs/human_prototype/{negatives}/build_loop_universe.log"
    conda:
        "../../../env/human_prep.yaml"
    shell:
        """
        PYTHONPATH=src/py python src/py/human_prepare_data/build_loop_universe.py \
            --raw_dir {config[raw_dir]} \
            --negatives {wildcards.negatives} {params.n2_args} \
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


# chromVAR ONCE on the analysis nuclei -> one shared motif x cell feature set.
# Every negative mode and feature set reuses this matrix (a cell group just
# selects its own cells at Stage 0), so the same cell is never scored twice
# under different peak filters / backgrounds.
rule compute_chromvar:
    input:
        snap=config["snap"],
        genome=config["genome"],
        genome_fai=f"{config['genome']}.fai",
        motifs=config["jaspar"],
        metadata="data/human_prototype/interim/cell_metadata.tsv",
    output:
        "data/human_prototype/interim/motifs_chromvar.tsv"
    threads: config["blas_threads"]
    log:
        "logs/human_prototype/compute_chromvar.log"
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


# Anchor x cell openness ONCE, per distinct anchor of the three conditions' real
# loops; both negative modes' loops pair these same anchors, so Stage 0 builds
# every loop universe's activity codes from this one matrix.
rule anchor_openness:
    input:
        snap=config["snap"],
        loop_beds=expand(f"{config['raw_dir']}/{{condition}}_loop.bed.gz", condition=config["conditions"]),
        metadata="data/human_prototype/interim/cell_metadata.tsv",
    output:
        "data/human_prototype/interim/anchor_openness.tsv"
    log:
        "logs/human_prototype/anchor_openness.log"
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


def partition_input(wildcards):
    """The feature set's cell-partition table, or nothing for pooled motif features."""
    spec = config["feature_sets"][wildcards.features]
    return [spec["partition"]] if spec else []


def partition_args(wildcards):
    """Stage 0 CLI arguments selecting the feature set's cell partition."""
    spec = config["feature_sets"][wildcards.features]
    return f"--partition {spec['partition']} --group_column {spec['column']}" if spec else ""


# Motif enrichment + per-condition loop-presence vectors, per negative mode and
# feature set.
rule run_stage0:
    input:
        openness="data/human_prototype/interim/anchor_openness.tsv",
        motifs="data/human_prototype/interim/motifs_chromvar.tsv",
        loop_table="data/human_prototype/interim/{negatives}/loop_universe.tsv",
        partition=partition_input,
    output:
        enrichment="results/human_prototype/{negatives}/prepare_data/{features}/motif_enrichment.csv",
        count11="results/human_prototype/{negatives}/prepare_data/{features}/count11.csv",
        n_cells="results/human_prototype/{negatives}/prepare_data/{features}/n_cells.tsv",
        y=expand("results/human_prototype/{{negatives}}/prepare_data/{{features}}/y_{cond}.csv",
                 cond=config["conditions"]),
    params:
        partition_args=partition_args,
    threads: config["stage0_threads"]
    log:
        "logs/human_prototype/{negatives}/stage0/{features}.log"
    conda:
        "../../../env/human_prep.yaml"
    shell:
        """
        export OPENBLAS_NUM_THREADS={threads} OMP_NUM_THREADS={threads} MKL_NUM_THREADS={threads}
        PYTHONPATH=src/py python src/py/human_prepare_data/run_stage0_human.py \
            --anchor_openness {input.openness} \
            --motifs {input.motifs} \
            --loop_table {input.loop_table} {params.partition_args} \
            --output_dir results/human_prototype/{wildcards.negatives}/prepare_data/{wildcards.features} \
            --log_path {log}
        """
