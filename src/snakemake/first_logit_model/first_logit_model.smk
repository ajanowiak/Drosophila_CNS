# Stage 3 - first logit model: preselects SHAP-ranked motifs (by EPV budget
# or by binary search) and fits a statsmodels Logit to get coefficient
# p-values for Stage 4's significant-feature selection.
#
# feature_selection_mode is split into two rules per value (binsearch vs.
# epv) rather than one wildcard-driven rule: epv needs an extra required
# --epv argument and its own wildcard, and the two output layouts otherwise
# only differ by <selection_tag>.
#
# shap_model only ranges over SHAP_MODELS (RF, XGB, defined in main.smk)
# since regression_coefs.py needs a SHAP ranking as its starting point.

rule all_first_logit_model:
    input:
        first_logit_model_targets()


rule regression_coefs_binsearch:
    input:
        shap_table="results/shap/{shap_model}/{tissue}_shap_table_{shap_model}.csv",
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
        summary="results/regression_coefs/{shap_model}/binsearch/{tissue}_summary.csv",
        coeffs_pdf="results/figures/regression_coefs/{shap_model}/binsearch/{tissue}_coefficients.pdf",
        coeffs_png="results/figures/regression_coefs/{shap_model}/binsearch/{tissue}_coefficients.png",
        volcano_pdf="results/figures/regression_coefs/{shap_model}/binsearch/{tissue}_volcano.pdf",
        volcano_png="results/figures/regression_coefs/{shap_model}/binsearch/{tissue}_volcano.png",
    log:
        "logs/regression_coefs/{shap_model}_binsearch_{tissue}.log"
    conda:
        "../../../env/analysis.yaml"
    shell:
        """
        PYTHONPATH=src/py python src/py/first_logit_model/regression_coefs.py \
            --tissue {wildcards.tissue} \
            --shap_model {wildcards.shap_model} \
            --feature_selection_mode binsearch \
            --motif_annotations_path {input.motif_names} \
            --top_n_coeffs {config[logit][top_n_coeffs]} \
            --p_value_threshold {config[logit][p_value_threshold]} \
            --volcano_effect_threshold {config[logit][volcano_effect_threshold]} \
            --log_path {log}
        """


rule regression_coefs_epv:
    input:
        shap_table="results/shap/{shap_model}/{tissue}_shap_table_{shap_model}.csv",
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
        summary="results/regression_coefs/{shap_model}/epv_{epv}/{tissue}_summary.csv",
        coeffs_pdf="results/figures/regression_coefs/{shap_model}/epv_{epv}/{tissue}_coefficients.pdf",
        coeffs_png="results/figures/regression_coefs/{shap_model}/epv_{epv}/{tissue}_coefficients.png",
        volcano_pdf="results/figures/regression_coefs/{shap_model}/epv_{epv}/{tissue}_volcano.pdf",
        volcano_png="results/figures/regression_coefs/{shap_model}/epv_{epv}/{tissue}_volcano.png",
    log:
        "logs/regression_coefs/{shap_model}_epv_{epv}_{tissue}.log"
    conda:
        "../../../env/analysis.yaml"
    shell:
        """
        PYTHONPATH=src/py python src/py/first_logit_model/regression_coefs.py \
            --tissue {wildcards.tissue} \
            --shap_model {wildcards.shap_model} \
            --feature_selection_mode epv \
            --epv {wildcards.epv} \
            --motif_annotations_path {input.motif_names} \
            --top_n_coeffs {config[logit][top_n_coeffs]} \
            --p_value_threshold {config[logit][p_value_threshold]} \
            --volcano_effect_threshold {config[logit][volcano_effect_threshold]} \
            --log_path {log}
        """


rule logit_regression_aucroc_binsearch:
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
    output:
        "results/logit_regression_cv_aucroc/{shap_model}/{tissue}_logit_cv_results_binsearch.csv"
    log:
        "logs/logit_regression_aucroc/{shap_model}_binsearch_{tissue}.log"
    conda:
        "../../../env/analysis.yaml"
    shell:
        """
        PYTHONPATH=src/py python src/py/first_logit_model/logit_regression_aucroc.py \
            --tissue {wildcards.tissue} \
            --shap_model {wildcards.shap_model} \
            --feature_selection_mode binsearch \
            --n_splits {config[n_splits]} \
            --p_value_threshold {config[logit][p_value_threshold]} \
            --log_path {log}
        """


rule logit_regression_aucroc_epv:
    input:
        summaries=expand(
            "results/regression_coefs/{{shap_model}}/epv_{epv}/{{tissue}}_summary.csv",
            epv=config["epv_values"],
        ),
        enrichment=expand(
            "results/training_data/unfiltered/hrs{window}/motif_enrichment_hrs{window}.csv",
            window=config["windows"],
        ),
        y=expand(
            "results/training_data/unfiltered/hrs{window}/y_{{tissue}}.csv",
            window=config["windows"],
        ),
    output:
        "results/logit_regression_cv_aucroc/{shap_model}/{tissue}_logit_cv_results_epv_sweep.csv"
    log:
        "logs/logit_regression_aucroc/{shap_model}_epv_sweep_{tissue}.log"
    params:
        epv_values=" ".join(str(epv) for epv in config["epv_values"]),
    conda:
        "../../../env/analysis.yaml"
    shell:
        """
        PYTHONPATH=src/py python src/py/first_logit_model/logit_regression_aucroc.py \
            --tissue {wildcards.tissue} \
            --shap_model {wildcards.shap_model} \
            --feature_selection_mode epv \
            --epv_values {params.epv_values} \
            --n_splits {config[n_splits]} \
            --p_value_threshold {config[logit][p_value_threshold]} \
            --log_path {log}
        """
