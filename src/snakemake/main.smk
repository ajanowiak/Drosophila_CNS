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
        "results/prepare_data/{filtering_mode}/hrs{window}/.done",
        filtering_mode=FILTERING_MODES,
        window=config["windows"],
    ) + expand(
        "results/prepare_data/{filtering_mode}/hrs{window}/y_{tissue}.csv",
        filtering_mode=FILTERING_MODES,
        window=config["windows"],
        tissue=config["tissues"],
    )


def train_full_models_targets():
    return (
        expand(
            "results/train_full_models/train_time_agnostic/models/{model}/{feature_mode}/{tissue}.pkl",
            model=config["models"],
            feature_mode=config["feature_modes"],
            tissue=config["tissues"],
        )
        + expand(
            "results/train_full_models/train_time_specific/models/all_data/{model}/hrs{window}/{tissue}.pkl",
            model=config["models"],
            window=config["windows"],
            tissue=config["tissues"],
        )
        + expand(
            "results/train_full_models/combined_roc_plot/figures/{model}/hrs{window}/combined.png",
            model=config["models"],
            window=config["windows"],
        )
        + expand(
            "results/train_full_models/compare_bar_plots/figures/{model}/{comparison}.png",
            model=config["models"],
            comparison=config["comparisons"],
        )
    )


def shap_importance_targets():
    return expand(
        "results/shap_importance/shap_analysis/tables/{model}/{tissue}.csv",
        model=SHAP_MODELS,
        tissue=config["tissues"],
    )


def first_logit_model_targets():
    return (
        expand(
            "results/first_logit_model/regression_coefs/tables/{shap_model}/binsearch/{tissue}.csv",
            shap_model=SHAP_MODELS,
            tissue=config["tissues"],
        )
        + expand(
            "results/first_logit_model/regression_coefs/tables/{shap_model}/epv_{epv}/{tissue}.csv",
            shap_model=SHAP_MODELS,
            tissue=config["tissues"],
            epv=config["epv_values"],
        )
        + expand(
            "results/first_logit_model/logit_regression_aucroc/tables/{shap_model}/binsearch/{tissue}.csv",
            shap_model=SHAP_MODELS,
            tissue=config["tissues"],
        )
        + expand(
            "results/first_logit_model/logit_regression_aucroc/tables/{shap_model}/epv_sweep/{tissue}.csv",
            shap_model=SHAP_MODELS,
            tissue=config["tissues"],
        )
    )


def second_logit_model_targets():
    return expand(
        "results/second_logit_model/logit_regression_significant_features/tables/{shap_model}/binsearch/{tissue}_summary.csv",
        shap_model=SHAP_MODELS,
        tissue=config["tissues"],
    ) + expand(
        "results/second_logit_model/logit_regression_significant_features/tables/{shap_model}/epv_{epv}/{tissue}_summary.csv",
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
