# Stage 2 - SHAP importance for each time-agnostic (curr feature mode)
# classifier trained in Stage 1. Feeds the first logit model's SHAP-ranked
# feature preselection.
#
# Only TreeExplainer models (RF, XGB, per SHAP_MODELS in main.smk) are wired
# into the default scope - KernelExplainer (LR, SVM) is computationally
# infeasible at this scale and was never completed by the old pipeline
# either. shap_analysis.py itself still accepts --model LR/SVM as an
# explicit target.

rule all_shap_importance:
    input:
        shap_importance_targets()


rule shap_analysis:
    input:
        model="results/train_full_models/train_time_agnostic/models/{model}/curr/{tissue}.pkl",
        enrichment=expand(
            "results/prepare_data/unfiltered/hrs{window}/motif_enrichment.csv",
            window=config["windows"],
        ),
        y=expand(
            "results/prepare_data/unfiltered/hrs{window}/y_{{tissue}}.csv",
            window=config["windows"],
        ),
        motif_names="data/motif_names.tsv",
    output:
        beeswarm_pdf="results/shap_importance/shap_analysis/figures/{model}/{tissue}.pdf",
        beeswarm_png="results/shap_importance/shap_analysis/figures/{model}/{tissue}.png",
        table="results/shap_importance/shap_analysis/tables/{model}/{tissue}.csv",
    log:
        "logs/shap_analysis/{model}_{tissue}.log"
    conda:
        "../../../env/analysis.yaml"
    shell:
        """
        PYTHONPATH=src/py python src/py/shap_importance/shap_analysis.py \
            --model {wildcards.model} \
            --tissue {wildcards.tissue} \
            --motif_annotations_path {input.motif_names} \
            --log_path {log}
        """
