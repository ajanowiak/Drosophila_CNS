# human_prepare_data/constants.py

"""Shared constants for the human-brain loop-prediction prototype."""

# The three Rahman et al. 2023 Hi-C conditions, in a fixed order.
CONDITIONS = ["fetal", "glia", "neuron"]

# How a condition's negative loops are drawn (see build_loop_universe.py):
#   other_conditions - the other conditions' loops whose Hi-C rectangle coincides
#                      with no real loop of this condition (disjoint rectangles);
#   n2               - Originally called "new_pairs_of_anchors": random re-pairing of this
#                      condition's own loop anchors.
NEGATIVE_MODES = ["other_conditions", "n2"]

# The ATAC cell sets whose enrichment we contrast.
CELL_SETS = ["neuron", "non_neuron", "all"]

# Atlas coarse classes: which ones are neurons, and the non-neuronal class.
NEURONAL_CLASSES = {"GLUT", "GABA"}
NON_NEURONAL_CLASS = "NonN"

# Columns pulled from the Li 2023 atlas metatable (Table S3).
ATLAS_COLUMNS = ["barcode", "sample", "cellclass", "subclass", "celltype"]

# Autosomes plus chrX; chromVAR ignores everything else (chrY, contigs, chrM).
MAIN_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX"]

# How each cell set is spelled in plot titles. "annotated non-neuron" keeps
# these atlas-labelled NonN nuclei distinct from the unannotated ones.
CELL_DISPLAY = {
    "neuron": "neuron",
    "non_neuron": "annotated non-neuron",
    "all": "all",
}

# Models that accept NaN features natively, so a loop with no both-open cell
# keeps its (NaN) enrichment row instead of being dropped.
NAN_TOLERANT_MODELS = {"XGB"}
