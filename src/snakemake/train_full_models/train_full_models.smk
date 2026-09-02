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
        f"results/prepare_data/unfiltered/hrs{w}/y_{wildcards.tissue}.csv"
        for w in config["windows"]
    ]

    if wildcards.feature_mode in ("curr", "expanded"):
        inputs += [
            f"results/prepare_data/unfiltered/hrs{w}/motif_enrichment.csv"
            for w in config["windows"]
        ]

    if wildcards.feature_mode in ("prev", "expanded"):
        inputs += [
            f"results/prepare_data/unfiltered/hrs{w}/motif_enrichment.csv"
            for w in config["windows_prev"]
        ]

    return inputs


rule train_time_agnostic:
    input:
        time_agnostic_inputs
    output:
        model="results/train_full_models/train_time_agnostic/models/{model}/{feature_mode}/{tissue}.pkl",
        roc_pdf="results/train_full_models/train_time_agnostic/figures/{model}/{feature_mode}/{tissue}.pdf",
        roc_png="results/train_full_models/train_time_agnostic/figures/{model}/{feature_mode}/{tissue}.png",
        summary="results/train_full_models/train_time_agnostic/cv_aucroc_summary/{model}/{feature_mode}/{tissue}.csv",
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
        model_cv="results/train_full_models/train_time_specific/models/cv/{model}/hrs{window}/{tissue}.pkl",
        model_all="results/train_full_models/train_time_specific/models/all_data/{model}/hrs{window}/{tissue}.pkl",
        roc_png="results/train_full_models/train_time_specific/figures/{model}/hrs{window}/{tissue}.png",
        roc_pdf="results/train_full_models/train_time_specific/figures/{model}/hrs{window}/{tissue}.pdf",
        roc_result="results/train_full_models/train_time_specific/roc_results/{model}/hrs{window}/{tissue}.pkl",
        summary="results/train_full_models/train_time_specific/cv_aucroc_summary/{model}/hrs{window}/{tissue}.csv",
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
            "results/train_full_models/train_time_specific/roc_results/{{model}}/hrs{{window}}/{tissue}.pkl",
            tissue=config["tissues"],
        )
    output:
        png="results/train_full_models/combined_roc_plot/figures/{model}/hrs{window}/combined.png",
        pdf="results/train_full_models/combined_roc_plot/figures/{model}/hrs{window}/combined.pdf",
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
            f"results/train_full_models/train_time_specific/cv_aucroc_summary/{model}/hrs{w}/{t}.csv"
            for w in config["windows"]
            for t in config["tissues"]
        ]
        inputs += [
            f"results/train_full_models/train_time_agnostic/cv_aucroc_summary/{model}/curr/{t}.csv"
            for t in config["tissues"]
        ]
        return inputs

    return [
        f"results/train_full_models/train_time_agnostic/cv_aucroc_summary/{model}/{mode}/{t}.csv"
        for mode in config["feature_modes"]
        for t in config["tissues"]
    ]


rule compare_bar_plots:
    input:
        compare_bar_plots_inputs
    output:
        png="results/train_full_models/compare_bar_plots/figures/{model}/{comparison}.png",
        pdf="results/train_full_models/compare_bar_plots/figures/{model}/{comparison}.pdf",
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
