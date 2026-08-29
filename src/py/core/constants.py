# constants.py
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

TISSUES = ["Neuroblasts", "Neurons", "Glia"]

# Short code -> classifier class, display name, and default hyperparameters.
# Short codes here must match config.yml's `models:` list, which is what
# Snakemake wildcards over.
MODELS = {
    "RF": {
        "class": RandomForestClassifier,
        "full": "Random Forest",
        "params": {"n_estimators": 500, "random_state": 0, "n_jobs": -1},
    },
    "SVM": {
        "class": SVC,
        "full": "Support Vector Machine",
        "params": {"probability": True},
    },
    "LR": {
        "class": LogisticRegression,
        "full": "Logistic Regression",
        "params": {"max_iter": 1000},
    },
    "XGB": {
        "class": XGBClassifier,
        "full": "XGBoost",
        "params": {"n_estimators": 500, "n_jobs": -1},
    },
}

# Reverse lookup for scripts that only have a loaded model object on hand
# (e.g. shap_analysis.py deriving a short code from type(model).__name__).
SKLEARN_CLASS_TO_SHORT = {v["class"].__name__: k for k, v in MODELS.items()}

# Enums
class FilteringMode(Enum):
    UNFILTERED = "unfiltered"
    NEURAL_LABELS = "neural_labels"
    REFINED_ANNOTATIONS = "refined_annotations"
