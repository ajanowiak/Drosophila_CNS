configfile: "config/config.yml"

# Must be a subset of the values of core.constants.FilteringMode.
# Only "unfiltered" is run by default -- "neural_labels" and
# "refined_annotations" are out of scope for this pass, even though
# compute_motif_enrichment.py still supports them.
FILTERING_MODES = ["unfiltered"]

# Only TreeExplainer models (RF, XGB) are wired into the default scope --
# KernelExplainer (LR, SVM) is computationally infeasible at this scale and
# was never completed by the old pipeline either. shap_analysis.py itself
# still accepts --model LR/SVM as an explicit target.
SHAP_MODELS = ["RF", "XGB"]


# One target-list function per stage, defined here (before any include) so
# both the master `rule all` below and that stage's own `all_<stage>` rule
# (in <stage>/<stage>.smk) share a single definition instead of two copies
# that could drift apart.

def prepare_data_targets():
    return expand(
        "results/training_data/{filtering_mode}/hrs{window}/.done",
        filtering_mode=FILTERING_MODES,
        window=config["windows"],
    ) + expand(
        "results/training_data/{filtering_mode}/hrs{window}/y_{tissue}.csv",
        filtering_mode=FILTERING_MODES,
        window=config["windows"],
        tissue=config["tissues"],
    )


def train_full_models_targets():
    return (
        expand(
            "results/models/time_agnostic/{model}/{feature_mode}/{model}_{tissue}_{feature_mode}.pkl",
            model=config["models"],
            feature_mode=config["feature_modes"],
            tissue=config["tissues"],
        )
        + expand(
            "results/models/time_specific/all_data/{model}/hrs{window}/{model}_{tissue}_hrs{window}.pkl",
            model=config["models"],
            window=config["windows"],
            tissue=config["tissues"],
        )
        + expand(
            "results/figures/time_specific/{model}/hrs{window}/roc_{model}_combined_hrs{window}.png",
            model=config["models"],
            window=config["windows"],
        )
        + expand(
            "results/figures/expanded_bar_plots/{model}/{comparison}_{model}.png",
            model=config["models"],
            comparison=config["comparisons"],
        )
    )


def shap_importance_targets():
    return expand(
        "results/shap/{model}/{tissue}_shap_table_{model}.csv",
        model=SHAP_MODELS,
        tissue=config["tissues"],
    )


def first_logit_model_targets():
    return (
        expand(
            "results/regression_coefs/{shap_model}/binsearch/{tissue}_summary.csv",
            shap_model=SHAP_MODELS,
            tissue=config["tissues"],
        )
        + expand(
            "results/regression_coefs/{shap_model}/epv_{epv}/{tissue}_summary.csv",
            shap_model=SHAP_MODELS,
            tissue=config["tissues"],
            epv=config["epv_values"],
        )
        + expand(
            "results/logit_regression_cv_aucroc/{shap_model}/{tissue}_logit_cv_results_binsearch.csv",
            shap_model=SHAP_MODELS,
            tissue=config["tissues"],
        )
        + expand(
            "results/logit_regression_cv_aucroc/{shap_model}/{tissue}_logit_cv_results_epv_sweep.csv",
            shap_model=SHAP_MODELS,
            tissue=config["tissues"],
        )
    )


def second_logit_model_targets():
    return expand(
        "results/second_logit_model/{shap_model}/binsearch/{tissue}_summary.csv",
        shap_model=SHAP_MODELS,
        tissue=config["tissues"],
    ) + expand(
        "results/second_logit_model/{shap_model}/epv_{epv}/{tissue}_summary.csv",
        shap_model=SHAP_MODELS,
        tissue=config["tissues"],
        epv=config["epv_values"],
    )


# rule all must stay the first rule in the workflow (across all includes) so
# it remains Snakemake's default target when none is given on the CLI.
rule all:
    input:
        prepare_data_targets()
        + train_full_models_targets()
        + shap_importance_targets()
        + first_logit_model_targets()
        + second_logit_model_targets(),


# One subdirectory per pipeline stage, mirroring src/py/'s layout.
include: "prepare_data/prepare_data.smk"
include: "train_full_models/train_full_models.smk"
include: "shap_importance/shap_importance.smk"
include: "first_logit_model/first_logit_model.smk"
include: "second_logit_model/second_logit_model.smk"
