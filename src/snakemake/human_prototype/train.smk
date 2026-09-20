# Stage 1 (human) - condition-specific XGBoost classifiers per ATAC cell set,
# then the (cell set x loop label) ROC-AUC grid across all of them.

rule all_train:
    input:
        train_targets()


# One cell set -> three condition classifiers + ROC figures + a CV summary.
rule run_stage1:
    input:
        enrichment="results/human_prototype/prepare_data/{cell_set}/motif_enrichment.csv",
        n_cells="results/human_prototype/prepare_data/{cell_set}/n_cells.txt",
        y=expand("results/human_prototype/prepare_data/{{cell_set}}/y_{cond}.csv",
                 cond=config["conditions"]),
    output:
        summary="results/human_prototype/train/{cell_set}/cv_summary.csv",
        combined="results/human_prototype/train/{cell_set}/figures/roc_combined.png",
        roc=expand("results/human_prototype/train/{{cell_set}}/figures/roc_{cond}.png",
                   cond=config["conditions"]),
    log:
        "logs/human_prototype/stage1/{cell_set}.log"
    conda:
        "../../../env/human_prep.yaml"
    shell:
        """
        PYTHONPATH=src/py python src/py/human_prepare_data/run_stage1_human.py \
            --stage0_dir results/human_prototype/prepare_data/{wildcards.cell_set} \
            --output_dir results/human_prototype/train/{wildcards.cell_set} \
            --cell_set {wildcards.cell_set} \
            --model {config[model]} \
            --n_splits {config[n_splits]} \
            --log_path {log}
        """


def aggregate_inputs(wildcards):
    """Every cell set's CV summary must exist before the grid is collated."""
    return expand(
        "results/human_prototype/train/{cell_set}/cv_summary.csv",
        cell_set=config["cell_sets"],
    )


rule aggregate_results:
    input:
        aggregate_inputs
    output:
        "results/human_prototype/train/all_cellsets_summary.csv"
    params:
        cell_sets=" ".join(config["cell_sets"]),
    log:
        "logs/human_prototype/aggregate.log"
    conda:
        "../../../env/human_prep.yaml"
    shell:
        """
        PYTHONPATH=src/py python src/py/human_prepare_data/aggregate_results.py \
            --train_dir results/human_prototype/train \
            --cell_sets {params.cell_sets} \
            --output {output} \
            --log_path {log}
        """
