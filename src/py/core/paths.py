# core/paths.py

"""
Centralized results/ path builders.

Pipeline context: results/ mirrors src/py/'s layout - one top-level
directory per stage, subdivided by script/rule, each with its own
figures/ subdirectory. Every path under results/ is built here once so
stage io.py modules and scripts share a single source of truth instead of
re-deriving the same literals independently.
"""

RESULTS_DIR = "results"


# prepare_data
#
# compute_motif_enrichment.py and extract_loop_presence_vectors.py share one
# directory per (filtering_mode, window) rather than getting separate
# per-rule subdirectories, because core.features.stack_windows() reads both
# rules' outputs together as a single training dataset for that window.

def prepare_data_dir(filtering_mode: str) -> str:
    return f"{RESULTS_DIR}/prepare_data/{filtering_mode}"


def prepare_data_window_dir(filtering_mode: str, window: str) -> str:
    return f"{prepare_data_dir(filtering_mode)}/hrs{window}"


# train_full_models

def train_time_agnostic_model_path(model: str, feature_mode: str, tissue: str) -> str:
    return f"{RESULTS_DIR}/train_full_models/train_time_agnostic/models/{model}/{feature_mode}/{tissue}.pkl"


def train_time_agnostic_summary_path(model: str, feature_mode: str, tissue: str) -> str:
    return f"{RESULTS_DIR}/train_full_models/train_time_agnostic/cv_aucroc_summary/{model}/{feature_mode}/{tissue}.csv"


def train_time_agnostic_figure_dir(model: str, feature_mode: str) -> str:
    return f"{RESULTS_DIR}/train_full_models/train_time_agnostic/figures/{model}/{feature_mode}"


def train_time_specific_model_path(split: str, model: str, window: str, tissue: str) -> str:
    """split is "cv" (best-fold model) or "all_data" (fit on the full dataset)."""
    return f"{RESULTS_DIR}/train_full_models/train_time_specific/models/{split}/{model}/hrs{window}/{tissue}.pkl"


def train_time_specific_summary_path(model: str, window: str, tissue: str) -> str:
    return f"{RESULTS_DIR}/train_full_models/train_time_specific/cv_aucroc_summary/{model}/hrs{window}/{tissue}.csv"


def train_time_specific_roc_result_path(model: str, window: str, tissue: str) -> str:
    return f"{RESULTS_DIR}/train_full_models/train_time_specific/roc_results/{model}/hrs{window}/{tissue}.pkl"


def train_time_specific_figure_dir(model: str, window: str) -> str:
    return f"{RESULTS_DIR}/train_full_models/train_time_specific/figures/{model}/hrs{window}"


def combined_roc_plot_figure_dir(model: str, window: str) -> str:
    return f"{RESULTS_DIR}/train_full_models/combined_roc_plot/figures/{model}/hrs{window}"


def compare_bar_plots_figure_dir(model: str) -> str:
    return f"{RESULTS_DIR}/train_full_models/compare_bar_plots/figures/{model}"


# shap_importance

def shap_table_path(model: str, tissue: str) -> str:
    return f"{RESULTS_DIR}/shap_importance/shap_analysis/tables/{model}/{tissue}.csv"


def shap_figure_dir(model: str) -> str:
    return f"{RESULTS_DIR}/shap_importance/shap_analysis/figures/{model}"


# first_logit_model

def regression_coefs_table_path(shap_model: str, selection_tag: str, tissue: str) -> str:
    return f"{RESULTS_DIR}/first_logit_model/regression_coefs/tables/{shap_model}/{selection_tag}/{tissue}.csv"


def regression_coefs_figure_dir(shap_model: str, selection_tag: str) -> str:
    return f"{RESULTS_DIR}/first_logit_model/regression_coefs/figures/{shap_model}/{selection_tag}"


def logit_regression_aucroc_table_path(shap_model: str, mode_tag: str, tissue: str) -> str:
    """mode_tag is "binsearch" (single-row) or "epv_sweep" (multi-row, one per epv value) - the run mode, not a specific selection_tag."""
    return f"{RESULTS_DIR}/first_logit_model/logit_regression_aucroc/tables/{shap_model}/{mode_tag}/{tissue}.csv"


# second_logit_model

def second_logit_model_table_dir(shap_model: str, selection_tag: str) -> str:
    return f"{RESULTS_DIR}/second_logit_model/logit_regression_significant_features/tables/{shap_model}/{selection_tag}"


def second_logit_model_figure_dir(shap_model: str, selection_tag: str) -> str:
    return f"{RESULTS_DIR}/second_logit_model/logit_regression_significant_features/figures/{shap_model}/{selection_tag}"
