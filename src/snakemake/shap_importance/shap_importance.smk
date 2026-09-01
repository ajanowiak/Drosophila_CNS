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
        model="results/models/time_agnostic/{model}/curr/{model}_{tissue}_curr.pkl",
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
        beeswarm_pdf="results/figures/shap/{model}/{tissue}_beeswarm_{model}.pdf",
        beeswarm_png="results/figures/shap/{model}/{tissue}_beeswarm_{model}.png",
        table="results/shap/{model}/{tissue}_shap_table_{model}.csv",
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
