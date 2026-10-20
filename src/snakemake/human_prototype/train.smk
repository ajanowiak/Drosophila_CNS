# Stage 1 (human) - one XGBoost classifier per negative mode x feature set x
# condition, a per-feature-set summary, then the ROC-AUC grid across all of them.

rule all_train:
    input:
        train_targets()


# One (negative mode, feature set, condition) -> one classifier + its ROC figure
# and summary row; the conditions run as separate, parallel jobs.
rule run_stage1:
    input:
        enrichment="results/human_prototype/{negatives}/prepare_data/{features}/motif_enrichment.csv",
        n_cells="results/human_prototype/{negatives}/prepare_data/{features}/n_cells.tsv",
        y="results/human_prototype/{negatives}/prepare_data/{features}/y_{condition}.csv",
    output:
        row="results/human_prototype/{negatives}/train/{features}/cv_{condition}.csv",
        cv_result="results/human_prototype/{negatives}/train/{features}/roc_results/{condition}.pkl",
        roc="results/human_prototype/{negatives}/train/{features}/figures/roc_{condition}.png",
    wildcard_constraints:
        condition="|".join(config["conditions"]),
    threads: config["stage1_threads"]
    log:
        "logs/human_prototype/{negatives}/stage1/{features}/{condition}.log"
    conda:
        "../../../env/human_prep.yaml"
    shell:
        """
        PYTHONPATH=src/py python src/py/human_prepare_data/run_stage1_human.py \
            --stage0_dir results/human_prototype/{wildcards.negatives}/prepare_data/{wildcards.features} \
            --output_dir results/human_prototype/{wildcards.negatives}/train/{wildcards.features} \
            --feature_set {wildcards.features} \
            --condition {wildcards.condition} \
            --model {config[model]} \
            --n_splits {config[n_splits]} \
            --n_jobs {threads} \
            --log_path {log}
        """


# One (negative mode, feature set): CV summary table + combined ROC of all conditions.
rule summarize_stage1:
    input:
        rows=expand("results/human_prototype/{{negatives}}/train/{{features}}/cv_{condition}.csv",
                    condition=config["conditions"]),
        cv_results=expand("results/human_prototype/{{negatives}}/train/{{features}}/roc_results/{condition}.pkl",
                          condition=config["conditions"]),
    output:
        summary="results/human_prototype/{negatives}/train/{features}/cv_summary.csv",
        combined="results/human_prototype/{negatives}/train/{features}/figures/roc_combined.png",
    params:
        conditions=" ".join(config["conditions"]),
    log:
        "logs/human_prototype/{negatives}/summarize_stage1/{features}.log"
    conda:
        "../../../env/human_prep.yaml"
    shell:
        """
        PYTHONPATH=src/py python src/py/human_prepare_data/summarize_stage1.py \
            --train_dir results/human_prototype/{wildcards.negatives}/train/{wildcards.features} \
            --feature_set {wildcards.features} \
            --conditions {params.conditions} \
            --log_path {log}
        """


# Every (negative mode x feature set) CV summary collated into one table.
rule aggregate_results:
    input:
        expand("results/human_prototype/{negatives}/train/{features}/cv_summary.csv",
               negatives=config["negatives"], features=FEATURE_SETS)
    output:
        "results/human_prototype/train_summary.csv"
    params:
        negatives=" ".join(config["negatives"]),
        feature_sets=" ".join(FEATURE_SETS),
    log:
        "logs/human_prototype/aggregate.log"
    conda:
        "../../../env/human_prep.yaml"
    shell:
        """
        PYTHONPATH=src/py python src/py/human_prepare_data/aggregate_results.py \
            --results_dir results/human_prototype \
            --negatives {params.negatives} \
            --feature_sets {params.feature_sets} \
            --output {output} \
            --log_path {log}
        """
