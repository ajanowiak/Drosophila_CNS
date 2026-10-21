# Logit (human) - a statsmodels Logit on the full feature set, per negative mode
# x feature set x condition. Runs in env/analysis.yaml, which has statsmodels.
# stderr is appended to the job log, so a failed fit's traceback and statsmodels'
# convergence / Hessian warnings end up next to the script's own log lines.

rule all_logit:
    input:
        logit_targets()


rule run_logit:
    input:
        enrichment="results/human_prototype/{negatives}/prepare_data/{features}/motif_enrichment.csv",
        y="results/human_prototype/{negatives}/prepare_data/{features}/y_{condition}.csv",
    output:
        summary="results/human_prototype/{negatives}/logit/{features}/summary_{condition}.csv",
        coefficients="results/human_prototype/{negatives}/logit/{features}/figures/{condition}_coefficients.png",
        volcano="results/human_prototype/{negatives}/logit/{features}/figures/{condition}_volcano.png",
    wildcard_constraints:
        condition="|".join(config["conditions"]),
    threads: config["logit_threads"]
    log:
        "logs/human_prototype/{negatives}/logit/{features}/{condition}.log"
    conda:
        "../../../env/analysis.yaml"
    shell:
        """
        export OPENBLAS_NUM_THREADS={threads} OMP_NUM_THREADS={threads} MKL_NUM_THREADS={threads}
        PYTHONPATH=src/py python src/py/human_prepare_data/run_logit_human.py \
            --stage0_dir results/human_prototype/{wildcards.negatives}/prepare_data/{wildcards.features} \
            --output_dir results/human_prototype/{wildcards.negatives}/logit/{wildcards.features} \
            --condition {wildcards.condition} \
            --p_value_threshold {config[logit][p_value_threshold]} \
            --top_n_coeffs {config[logit][top_n_coeffs]} \
            --volcano_effect_threshold {config[logit][volcano_effect_threshold]} \
            --log_path {log} \
            2>> {log}
        """
