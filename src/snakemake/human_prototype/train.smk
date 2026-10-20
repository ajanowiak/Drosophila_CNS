# Stage 1 (human) - condition-specific XGBoost classifiers per negative mode and
# feature set, then the ROC-AUC grid across all of them.

rule all_train:
    input:
        train_targets()


# One (negative mode, feature set) -> three condition classifiers + ROC figures + a CV summary.
rule run_stage1:
    input:
        enrichment="results/human_prototype/{negatives}/prepare_data/{features}/motif_enrichment.csv",
        n_cells="results/human_prototype/{negatives}/prepare_data/{features}/n_cells.tsv",
        y=expand("results/human_prototype/{{negatives}}/prepare_data/{{features}}/y_{cond}.csv",
                 cond=config["conditions"]),
    output:
        summary="results/human_prototype/{negatives}/train/{features}/cv_summary.csv",
        combined="results/human_prototype/{negatives}/train/{features}/figures/roc_combined.png",
        roc=expand("results/human_prototype/{{negatives}}/train/{{features}}/figures/roc_{cond}.png",
                   cond=config["conditions"]),
    log:
        "logs/human_prototype/{negatives}/stage1/{features}.log"
    conda:
        "../../../env/human_prep.yaml"
    shell:
        """
        PYTHONPATH=src/py python src/py/human_prepare_data/run_stage1_human.py \
            --stage0_dir results/human_prototype/{wildcards.negatives}/prepare_data/{wildcards.features} \
            --output_dir results/human_prototype/{wildcards.negatives}/train/{wildcards.features} \
            --feature_set {wildcards.features} \
            --model {config[model]} \
            --n_splits {config[n_splits]} \
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
