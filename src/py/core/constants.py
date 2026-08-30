# constants.py

"""
Shared, cross-stage constants: pipeline dimensions (windows, tissues) and
the classifier registry.

Pipeline context: importable from every stage directory (00_prepare_data,
train_full_models, shap_importance, first_logit_model, second_logit_model)
via PYTHONPATH=src/py, so the same values never need to be redefined per
stage.
"""

from enum import Enum

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Constants

ACTIVITY_PROFILES = ["1-1", "1-0", "0-1", "0-0"]

NEURAL_LABELS_RAW = [
    "Brain", "Neural", "Ventral_nerve_cord",
    "Ventral_nerve_cord_prim", "Glia", "PNS_&_sense"
]

NEURAL_LABELS = ['Brain','Neural','Ventral nerve cord',
                 'Ventral nerve cord prim.','Glia','PNS & sense'
                 ]

WINDOWS = ["06-08", "10-12", "14-16"]
WINDOWS_PREV = ["04-06", "08-10", "12-14"]

# main window -> its immediate predecessor, derived from the two lists above
# so they can't drift apart
PREV_WINDOW = dict(zip(WINDOWS, WINDOWS_PREV))

TISSUES = ["Neuroblasts", "Neurons", "Glia"]

# short code -> classifier class and display name. short codes here must
# match config.yml's `models:` list, which is what Snakemake wildcards over.
# hyperparameters are NOT here: train_time_specific and train_time_agnostic
# use different hyperparameters for the same model names (see below), so
# there is no single canonical params dict.
MODELS = {
    "RF": {
        "class": RandomForestClassifier,
        "full": "Random Forest",
    },
    "SVM": {
        "class": SVC,
        "full": "Support Vector Machine",
    },
    "LR": {
        "class": LogisticRegression,
        "full": "Logistic Regression",
    },
    "XGB": {
        "class": XGBClassifier,
        "full": "XGBoost",
    },
}

# reverse lookup for scripts that only have a loaded model object on hand
# (e.g. shap_analysis.py deriving a short code from type(model).__name__)
SKLEARN_CLASS_TO_SHORT = {v["class"].__name__: k for k, v in MODELS.items()}

# hyperparameters as used by train_time_specific.py. Kept separate from
# TIME_AGNOSTIC_MODEL_PARAMS below because the two scripts were already
# using different values before this refactor -- preserved as-is rather
# than reconciled, since reconciling would change trained-model results.
TIME_SPECIFIC_MODEL_PARAMS = {
    "RF": dict(n_estimators=500, random_state=0, n_jobs=-1),
    "SVM": dict(probability=True),
    "LR": dict(max_iter=1000),
    "XGB": dict(n_estimators=500, n_jobs=-1),
}

# hyperparameters as used by train_time_agnostic_expanded.py (now covers
# curr / prev / expanded feature modes).
TIME_AGNOSTIC_MODEL_PARAMS = {
    "RF": dict(n_estimators=500, random_state=0, n_jobs=4),
    "SVM": dict(probability=True, random_state=0),
    "LR": dict(max_iter=1000, random_state=0, n_jobs=4),
    "XGB": dict(n_estimators=500, random_state=0, n_jobs=4),
}

# Enums
class FilteringMode(Enum):
    UNFILTERED = "unfiltered"
    NEURAL_LABELS = "neural_labels"
    REFINED_ANNOTATIONS = "refined_annotations"


class FeatureMode(Enum):
    CURRENT = "curr"
    PREVIOUS = "prev"
    EXPANDED = "expanded"
