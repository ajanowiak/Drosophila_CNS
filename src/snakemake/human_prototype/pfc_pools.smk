# Prototype, standalone (not included in the main human Snakefile): pooled-cell
# models on the GSE244618 PFC libraries in config/human_pfc_libraries.yml (pfc.pools).
#
# Each pooled library first gets its per-loop chromVAR sums over "1-1" and other
# cells (library_loop_sums). Pooled Stage 0 adds them up over the libraries of a
# pool, which gives exactly the enrichment of their cells taken together:
#   motif_x_region - model (a): one pool per brain region, one feature per motif x region;
#   motif_pooled   - model (b): all libraries in one pool, one feature per motif.
# Every feature set is trained with the main pipeline's Stage 1 (XGB, CV), one job
# per condition.
#
# Per-library inputs (anchor openness, chromVAR) and the loop universe must already
# exist from pfc_libraries.smk and the main pipeline. Run from the repo root:
#
#   snakemake --snakefile src/snakemake/human_prototype/pfc_pools.smk --use-conda --cores 16

configfile: "config/human_prototype.yml"
configfile: "config/human_pfc_libraries.yml"

PFC = config["pfc"]
POOLS = PFC["pools"]
NEGATIVES = PFC["negatives"]
POOL_LIBRARIES = list(POOLS["libraries"])
FEATURE_SETS = list(POOLS["feature_sets"])

LOOP_TABLE = f"data/human_prototype/interim/{NEGATIVES}/loop_universe.tsv"
LIBRARY_INTERIM = "data/human_prototype/interim/libraries/{library}"
LOOP_SUMS = f"data/human_prototype/interim/{NEGATIVES}/libraries/{{library}}/loop_sums.npz"
POOL_RESULTS = f"results/human_prototype/{NEGATIVES}/pools"

wildcard_constraints:
    library="|".join(POOL_LIBRARIES),
    features="|".join(FEATURE_SETS),
    condition="|".join(config["conditions"]),


def pool_groups(wildcards):
    """--groups argument of a feature set: each pooled library's group, or nothing for one pool."""
    attribute = POOLS["feature_sets"][wildcards.features]
    if attribute is None:
        return ""
    return "--groups " + " ".join(PFC["libraries"][library][attribute] for library in POOL_LIBRARIES)


rule pools:
    input:
        expand(f"{POOL_RESULTS}/train/{{features}}/cv_summary.csv", features=FEATURE_SETS),


# Per-loop chromVAR sums over "1-1" and other cells, and anchor-open counts, of one library.
rule library_loop_sums:
    input:
        openness=f"{LIBRARY_INTERIM}/anchor_openness.tsv",
        motifs=f"{LIBRARY_INTERIM}/motifs_chromvar.tsv",
        loop_table=LOOP_TABLE,
    output:
        LOOP_SUMS
    threads: config["stage0_threads"]
    log:
        f"logs/human_prototype/{NEGATIVES}/libraries/{{library}}/loop_sums.log"
    conda:
        "../../../env/human_prep.yaml"
    shell:
        """
        export OPENBLAS_NUM_THREADS={threads} OMP_NUM_THREADS={threads} MKL_NUM_THREADS={threads}
        PYTHONPATH=src/py python src/py/human_prepare_data/compute_loop_sums.py \
            --anchor_openness {input.openness} \
            --motifs {input.motifs} \
            --loop_table {input.loop_table} \
            --output {output} \
            --log_path {log}
        """


# Motif enrichment of the pooled libraries' cells (models a and b).
rule pool_stage0:
    input:
        sums=expand(LOOP_SUMS, library=POOL_LIBRARIES),
        loop_table=LOOP_TABLE,
    output:
        enrichment=f"{POOL_RESULTS}/prepare_data/{{features}}/motif_enrichment.csv",
        count11=f"{POOL_RESULTS}/prepare_data/{{features}}/count11.csv",
        n_cells=f"{POOL_RESULTS}/prepare_data/{{features}}/n_cells.tsv",
        y=[f"{POOL_RESULTS}/prepare_data/{{features}}/y_{cond}.csv" for cond in config["conditions"]],
    params:
        groups=pool_groups,
        output_dir=lambda wildcards: f"{POOL_RESULTS}/prepare_data/{wildcards.features}",
    log:
        f"logs/human_prototype/{NEGATIVES}/pools/stage0/{{features}}.log"
    conda:
        "../../../env/human_prep.yaml"
    shell:
        """
        PYTHONPATH=src/py python src/py/human_prepare_data/run_stage0_pooled.py \
            --loop_sums {input.sums} {params.groups} \
            --loop_table {input.loop_table} \
            --output_dir {params.output_dir} \
            --log_path {log}
        """


# One (feature set, condition) -> one classifier, as rule run_stage1 of the main pipeline.
rule pool_stage1:
    input:
        enrichment=f"{POOL_RESULTS}/prepare_data/{{features}}/motif_enrichment.csv",
        n_cells=f"{POOL_RESULTS}/prepare_data/{{features}}/n_cells.tsv",
        y=f"{POOL_RESULTS}/prepare_data/{{features}}/y_{{condition}}.csv",
    output:
        row=f"{POOL_RESULTS}/train/{{features}}/cv_{{condition}}.csv",
        cv_result=f"{POOL_RESULTS}/train/{{features}}/roc_results/{{condition}}.pkl",
        roc=f"{POOL_RESULTS}/train/{{features}}/figures/roc_{{condition}}.png",
    params:
        stage0_dir=lambda wildcards: f"{POOL_RESULTS}/prepare_data/{wildcards.features}",
        output_dir=lambda wildcards: f"{POOL_RESULTS}/train/{wildcards.features}",
    threads: config["stage1_threads"]
    log:
        f"logs/human_prototype/{NEGATIVES}/pools/stage1/{{features}}/{{condition}}.log"
    conda:
        "../../../env/human_prep.yaml"
    shell:
        """
        export OPENBLAS_NUM_THREADS={threads} OMP_NUM_THREADS={threads} MKL_NUM_THREADS={threads}
        PYTHONPATH=src/py python src/py/human_prepare_data/run_stage1_human.py \
            --stage0_dir {params.stage0_dir} \
            --output_dir {params.output_dir} \
            --feature_set {wildcards.features} \
            --condition {wildcards.condition} \
            --model {config[model]} \
            --n_splits {config[n_splits]} \
            --n_jobs {threads} \
            --log_path {log}
        """


# One feature set: CV summary table + combined ROC of all conditions.
rule pool_summarize_stage1:
    input:
        rows=expand(f"{POOL_RESULTS}/train/{{{{features}}}}/cv_{{condition}}.csv", condition=config["conditions"]),
        cv_results=expand(f"{POOL_RESULTS}/train/{{{{features}}}}/roc_results/{{condition}}.pkl",
                          condition=config["conditions"]),
    output:
        summary=f"{POOL_RESULTS}/train/{{features}}/cv_summary.csv",
        combined=f"{POOL_RESULTS}/train/{{features}}/figures/roc_combined.png",
    params:
        conditions=" ".join(config["conditions"]),
        train_dir=lambda wildcards: f"{POOL_RESULTS}/train/{wildcards.features}",
    log:
        f"logs/human_prototype/{NEGATIVES}/pools/summarize_stage1/{{features}}.log"
    conda:
        "../../../env/human_prep.yaml"
    shell:
        """
        PYTHONPATH=src/py python src/py/human_prepare_data/summarize_stage1.py \
            --train_dir {params.train_dir} \
            --feature_set {wildcards.features} \
            --conditions {params.conditions} \
            --log_path {log}
        """
