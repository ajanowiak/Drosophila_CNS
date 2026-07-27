# constants.py
from enum import Enum

# Constants

WINDOWS = ['06-08', '10-12', '14-16']
WINDOWS_PREV = ["04-06", "08-10", "12-14"]

ACTIVITY_PROFILES = ["1-1", "1-0", "0-1", "0-0"]

NEURAL_LABELS_RAW = [
    "Brain", "Neural", "Ventral_nerve_cord",
    "Ventral_nerve_cord_prim", "Glia", "PNS_&_sense"
]

NEURAL_LABELS = ['Brain','Neural','Ventral nerve cord',
                 'Ventral nerve cord prim.','Glia','PNS & sense'
                 ]


# Enums
class FilteringMode(Enum):
    UNFILTERED = "unfiltered"
    NEURAL_LABELS = "neural_labels"
    REFINED_ANNOTATIONS = "refined_annotations"
