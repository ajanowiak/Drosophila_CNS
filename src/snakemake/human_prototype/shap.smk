# Stage 2 (human) - SHAP importance for one XGBoost classifier per negative mode x
# feature set x condition, as in the Drosophila shap_importance stage. Human Stage 1
# keeps no fitted model, so each job fits the all-data model itself (same loops and
# hyperparameters as Stage 1) and saves it next to the SHAP outputs.
#
# Runs in the analysis env (the Drosophila SHAP env), which has shap; human_prep does not.

rule all_shap:
    input:
        shap_targets()


rule run_shap:
    input:
        enrichment="results/human_prototype/{negatives}/prepare_data/{features}/motif_enrichment.csv",
        n_cells="results/human_prototype/{negatives}/prepare_data/{features}/n_cells.tsv",
        y="results/human_prototype/{negatives}/prepare_data/{features}/y_{condition}.csv",
    output:
        model="results/human_prototype/{negatives}/shap/{features}/models/{condition}.pkl",
        table="results/human_prototype/{negatives}/shap/{features}/tables/{condition}.csv",
        beeswarm_pdf="results/human_prototype/{negatives}/shap/{features}/figures/{condition}.pdf",
        beeswarm_png="results/human_prototype/{negatives}/shap/{features}/figures/{condition}.png",
    wildcard_constraints:
        condition="|".join(config["conditions"]),
    threads: config["shap_threads"]
    log:
        "logs/human_prototype/{negatives}/shap/{features}/{condition}.log"
    conda:
        "../../../env/analysis.yaml"
    shell:
        """
        export OPENBLAS_NUM_THREADS={threads} OMP_NUM_THREADS={threads} MKL_NUM_THREADS={threads}
        PYTHONPATH=src/py python src/py/human_prepare_data/run_shap_human.py \
            --stage0_dir results/human_prototype/{wildcards.negatives}/prepare_data/{wildcards.features} \
            --output_dir results/human_prototype/{wildcards.negatives}/shap/{wildcards.features} \
            --feature_set {wildcards.features} \
            --condition {wildcards.condition} \
            --model {config[model]} \
            --n_jobs {threads} \
            --log_path {log}
        """
