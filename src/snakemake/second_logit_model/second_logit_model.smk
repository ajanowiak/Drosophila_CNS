# Stage 4 - second logit model: retrains on only the BH-significant motifs
# from Stage 3, for both the epv and binsearch selections.
#
# shap_model only ranges over SHAP_MODELS (RF, XGB, defined in main.smk).

rule all_second_logit_model:
    input:
        second_logit_model_targets()


rule second_logit_model_binsearch:
    input:
        summary="results/regression_coefs/{shap_model}/binsearch/{tissue}_summary.csv",
        enrichment=expand(
            "results/training_data/unfiltered/hrs{window}/motif_enrichment_hrs{window}.csv",
            window=config["windows"],
        ),
        y=expand(
            "results/training_data/unfiltered/hrs{window}/y_{{tissue}}.csv",
            window=config["windows"],
        ),
        motif_names="data/motif_names.tsv",
    output:
        summary="results/second_logit_model/{shap_model}/binsearch/{tissue}_summary.csv",
        cv_results="results/second_logit_model/{shap_model}/binsearch/{tissue}_cv_results.csv",
        coeffs_pdf="results/figures/second_logit_model/{shap_model}/binsearch/{tissue}_coefficients.pdf",
        coeffs_png="results/figures/second_logit_model/{shap_model}/binsearch/{tissue}_coefficients.png",
        volcano_pdf="results/figures/second_logit_model/{shap_model}/binsearch/{tissue}_volcano.pdf",
        volcano_png="results/figures/second_logit_model/{shap_model}/binsearch/{tissue}_volcano.png",
    log:
        "logs/second_logit_model/{shap_model}_binsearch_{tissue}.log"
    conda:
        "../../../env/analysis.yaml"
    shell:
        """
        PYTHONPATH=src/py python src/py/second_logit_model/logit_regression_significant_features.py \
            --tissue {wildcards.tissue} \
            --shap_model {wildcards.shap_model} \
            --feature_selection_mode binsearch \
            --n_splits {config[n_splits]} \
            --p_value_threshold {config[logit][p_value_threshold]} \
            --top_n_coeffs {config[logit][top_n_coeffs]} \
            --volcano_effect_threshold {config[logit][volcano_effect_threshold]} \
            --motif_annotations_path {input.motif_names} \
            --log_path {log}
        """


rule second_logit_model_epv:
    input:
        summary="results/regression_coefs/{shap_model}/epv_{epv}/{tissue}_summary.csv",
        enrichment=expand(
            "results/training_data/unfiltered/hrs{window}/motif_enrichment_hrs{window}.csv",
            window=config["windows"],
        ),
        y=expand(
            "results/training_data/unfiltered/hrs{window}/y_{{tissue}}.csv",
            window=config["windows"],
        ),
        motif_names="data/motif_names.tsv",
    output:
        summary="results/second_logit_model/{shap_model}/epv_{epv}/{tissue}_summary.csv",
        cv_results="results/second_logit_model/{shap_model}/epv_{epv}/{tissue}_cv_results.csv",
        coeffs_pdf="results/figures/second_logit_model/{shap_model}/epv_{epv}/{tissue}_coefficients.pdf",
        coeffs_png="results/figures/second_logit_model/{shap_model}/epv_{epv}/{tissue}_coefficients.png",
        volcano_pdf="results/figures/second_logit_model/{shap_model}/epv_{epv}/{tissue}_volcano.pdf",
        volcano_png="results/figures/second_logit_model/{shap_model}/epv_{epv}/{tissue}_volcano.png",
    log:
        "logs/second_logit_model/{shap_model}_epv_{epv}_{tissue}.log"
    conda:
        "../../../env/analysis.yaml"
    shell:
        """
        PYTHONPATH=src/py python src/py/second_logit_model/logit_regression_significant_features.py \
            --tissue {wildcards.tissue} \
            --shap_model {wildcards.shap_model} \
            --feature_selection_mode epv \
            --epv {wildcards.epv} \
            --n_splits {config[n_splits]} \
            --p_value_threshold {config[logit][p_value_threshold]} \
            --top_n_coeffs {config[logit][top_n_coeffs]} \
            --volcano_effect_threshold {config[logit][volcano_effect_threshold]} \
            --motif_annotations_path {input.motif_names} \
            --log_path {log}
        """
