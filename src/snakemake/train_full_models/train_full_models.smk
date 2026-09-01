# Stage 1 - trains time-specific and time-agnostic (curr/prev/expanded)
# classifiers, plus the comparison plots built from their CV summaries.

rule all_train_full_models:
    input:
        train_full_models_targets()


def time_agnostic_inputs(wildcards):
    """
    Enrichment matrices needed for one time-agnostic training run.

    curr/expanded need the current window's own enrichment matrix; prev/
    expanded need the preceding window's. y labels are always the current
    window's, regardless of feature_mode (see core.features.stack_windows).
    """
    inputs = [
        f"results/training_data/unfiltered/hrs{w}/y_{wildcards.tissue}.csv"
        for w in config["windows"]
    ]

    if wildcards.feature_mode in ("curr", "expanded"):
        inputs += [
            f"results/training_data/unfiltered/hrs{w}/motif_enrichment_hrs{w}.csv"
            for w in config["windows"]
        ]

    if wildcards.feature_mode in ("prev", "expanded"):
        inputs += [
            f"results/training_data/unfiltered/hrs{w}/motif_enrichment_hrs{w}.csv"
            for w in config["windows_prev"]
        ]

    return inputs


rule train_time_agnostic:
    input:
        time_agnostic_inputs
    output:
        model="results/models/time_agnostic/{model}/{feature_mode}/{model}_{tissue}_{feature_mode}.pkl",
        roc_pdf="results/figures/time_agnostic/{model}_{feature_mode}/roc_{model}_{feature_mode}_{tissue}.pdf",
        roc_png="results/figures/time_agnostic/{model}_{feature_mode}/roc_{model}_{feature_mode}_{tissue}.png",
        summary="results/time_agnostic/{model}/{feature_mode}/cv_aucroc_summary_{model}_{feature_mode}_{tissue}.csv",
    log:
        "logs/train_time_agnostic/{model}_{feature_mode}_{tissue}.log"
    conda:
        "../../../env/analysis.yaml"
    shell:
        """
        PYTHONPATH=src/py python src/py/train_full_models/train_time_agnostic.py \
            --model {wildcards.model} \
            --tissue {wildcards.tissue} \
            --feature_mode {wildcards.feature_mode} \
            --n_splits {config[n_splits]} \
            --log_path {log}
        """


rule train_time_specific:
    input:
        X="data/training/hrs{window}/data_diff_hrs{window}.csv",
        y="data/training/hrs{window}/y_{tissue}.csv",
    output:
        model_cv="results/models/time_specific/cv/{model}/hrs{window}/{model}_{tissue}_hrs{window}.pkl",
        model_all="results/models/time_specific/all_data/{model}/hrs{window}/{model}_{tissue}_hrs{window}.pkl",
        roc_png="results/figures/time_specific/{model}/hrs{window}/roc_{model}_{tissue}_hrs{window}.png",
        roc_pdf="results/figures/time_specific/{model}/hrs{window}/roc_{model}_{tissue}_hrs{window}.pdf",
        roc_result="results/time_specific/{model}/hrs{window}/roc_result_{model}_hrs{window}_{tissue}.pkl",
        summary="results/time_specific/{model}/hrs{window}/cv_aucroc_summary_{model}_hrs{window}_{tissue}.csv",
    log:
        "logs/train_time_specific/{model}_hrs{window}_{tissue}.log"
    conda:
        "../../../env/analysis.yaml"
    shell:
        """
        PYTHONPATH=src/py python src/py/train_full_models/train_time_specific.py \
            --model {wildcards.model} \
            --tissue {wildcards.tissue} \
            --window {wildcards.window} \
            --n_splits {config[n_splits]} \
            --log_path {log}
        """


rule combined_roc_plot:
    input:
        expand(
            "results/time_specific/{{model}}/hrs{{window}}/roc_result_{{model}}_hrs{{window}}_{tissue}.pkl",
            tissue=config["tissues"],
        )
    output:
        png="results/figures/time_specific/{model}/hrs{window}/roc_{model}_combined_hrs{window}.png",
        pdf="results/figures/time_specific/{model}/hrs{window}/roc_{model}_combined_hrs{window}.pdf",
    log:
        "logs/combined_roc_plot/{model}_hrs{window}.log"
    conda:
        "../../../env/analysis.yaml"
    shell:
        """
        PYTHONPATH=src/py python src/py/train_full_models/combined_roc_plot.py \
            --model {wildcards.model} \
            --window {wildcards.window} \
            --log_path {log}
        """


def compare_bar_plots_inputs(wildcards):
    """
    CV summary CSVs one bar-chart comparison needs, for one model.

    time_specific_vs_agnostic compares every time-specific (window, tissue)
    summary against the time-agnostic "curr" summaries; curr_prev_expanded
    compares all three time-agnostic feature modes against each other.
    """
    model = wildcards.model

    if wildcards.comparison == "time_specific_vs_agnostic":
        inputs = [
            f"results/time_specific/{model}/hrs{w}/cv_aucroc_summary_{model}_hrs{w}_{t}.csv"
            for w in config["windows"]
            for t in config["tissues"]
        ]
        inputs += [
            f"results/time_agnostic/{model}/curr/cv_aucroc_summary_{model}_curr_{t}.csv"
            for t in config["tissues"]
        ]
        return inputs

    return [
        f"results/time_agnostic/{model}/{mode}/cv_aucroc_summary_{model}_{mode}_{t}.csv"
        for mode in config["feature_modes"]
        for t in config["tissues"]
    ]


rule compare_bar_plots:
    input:
        compare_bar_plots_inputs
    output:
        png="results/figures/expanded_bar_plots/{model}/{comparison}_{model}.png",
        pdf="results/figures/expanded_bar_plots/{model}/{comparison}_{model}.pdf",
    log:
        "logs/compare_bar_plots/{model}_{comparison}.log"
    conda:
        "../../../env/analysis.yaml"
    shell:
        """
        PYTHONPATH=src/py python src/py/train_full_models/compare_bar_plots.py \
            --model {wildcards.model} \
            --comparison {wildcards.comparison} \
            --log_path {log}
        """
